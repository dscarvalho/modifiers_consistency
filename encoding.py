import json
import logging
import torch
import numpy as np
import gensim.downloader as gensim_api
from typing import List, Tuple
from torch import Tensor, device as Device
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from dynaconf import settings
from data_access.adjectives_nouns import Adjectives, Nouns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

VECOPS = {
    "sum": torch.sum,
    "mul": torch.prod,
    "mean": lambda x: torch.mean(x, dim=0)
}

ENCODERS = {
    "dpr": ("sentence-transformers/facebook-dpr-ctx_encoder-multiset-base", None, "transformer"),
    "labse": ("sentence-transformers/LaBSE", None, "transformer"),
    "specter": ("sentence-transformers/allenai-specter", None, "transformer"),
    "nv-embed-v2": ("nvidia/NV-Embed-v2", None, "transformer"),
    "stella_en_1.5B_v5": ("dunzhang/stella_en_1.5B_v5", None, "transformer"),
    "text-embeddings-3-small": ("text-embeddings-3-small", None, "openai"),
    "glove": ("glove-wiki-gigaword-300", None, "gensim"),
    "word2vec": ("word2vec-google-news-300", None, "gensim")
}


def get_transformer_model(enc_name: Tuple[str], sentence: bool, device: Device):
    tokenizer = None
    model = None
    if (sentence):
        model = SentenceTransformer(enc_name[0], trust_remote_code=True).to(device)
    else:
        tokenizer = enc_name[4].from_pretrained(enc_name[1])
        model = enc_name[3].from_pretrained(enc_name[1]).to(device)

    return model, tokenizer


def phrase_from_index(idx_phrase: List[List[int]], synonym: bool = False, idx_syn: int = 0):
    decoded = list()
    adjectives = Adjectives()
    nouns = Nouns()
    i = 0
    for idx_word, idx_type in idx_phrase:
        if (idx_type == 0):
            if (not synonym or i != idx_syn):
                decoded.append(nouns.get_word(idx_word))
            else:
                decoded.append(nouns.get_synonym(idx_word))
        else:
            if (not synonym or i != idx_syn):
                decoded.append(adjectives.get_word(idx_word))
            else:
                decoded.append(adjectives.get_synonym(idx_word))

        i += 1

    return decoded


class EncoderNotSupportedException(Exception):
    def __init__(self, msg: str = "The encoder type is not supported."):
        self.message = msg


class GensimModel:
    def __init__(self, model_name: str, device: Device):
        self._model = gensim_api.load(model_name)
        self.device: Device = device

    def encode(self, phrases: List[str], convert_to_tensor=True) -> Tensor:
        dim = self._model["x"].shape
        embeddings = [np.mean([self._model[token] if (token in self._model) else np.zeros(dim) for token in phrase.split()], axis=0)
                      for phrase in phrases]

        return torch.stack([torch.tensor(emb).to(self.device) for emb in embeddings])


class OpenAIEmbeddingModel:
    call_count = 0

    def __init__(self, model_name: str, device: Device):
        with open(settings["openai_auth_file"]) as auth_file:
            auth_info = json.load(auth_file)
        self.client = AzureOpenAI(api_key=auth_info["api_key"],
                                  api_version=auth_info["api_version"],
                                  azure_endpoint=auth_info["endpoint"])
        self.model_name = auth_info["deployment_name"] if ("deployment_name" in auth_info) else model_name
        self.device: Device = device

    def encode(self, phrases: List[str], convert_to_tensor=True) -> Tensor:
        response = self.client.embeddings.create(input=phrases, model=self.model_name).data
        embeds = torch.tensor([emb.embedding for emb in response], device=self.device)
        OpenAIEmbeddingModel.call_count += 1
        logger.info(f"OpenAI embedding API calls: {OpenAIEmbeddingModel.call_count}")

        return embeds


class PhraseEncoder:
    _cache = dict()

    def __init__(self, model_name: str = "dpr", sentence: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._model = None
        self._sentence = sentence
        self._model_name = model_name
        enc_name = ENCODERS[model_name]

        if (enc_name[2] == "transformer"):
            self._model, self._tokenizer = get_transformer_model(enc_name, sentence, self.device)
        elif (enc_name[2] == "gensim"):
            self._model = GensimModel(enc_name[0], self.device)
        elif (enc_name[2] == "openai"):
            self._model = OpenAIEmbeddingModel(enc_name[0], self.device)
        else:
            raise EncoderNotSupportedException

    def encode(self, phrases: List[str], op: str = None, cache: bool = False) -> Tensor:
        cache_key = (tuple(phrases), self._model_name, self._sentence, op)
        if (cache and cache_key in PhraseEncoder._cache):
            enc = PhraseEncoder._cache[cache_key]
        elif (self._sentence or type(self._model) == GensimModel):
            enc = self._model.encode(phrases, convert_to_tensor=True)
        else:
            token_res = self._tokenizer(phrases, padding=True, return_tensors="pt")
            input_ids = token_res.input_ids.to(self.device)
            attn_mask = token_res.attention_mask.to(self.device)
            enc = self._model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

            for b_idx in range(enc.shape[0]):
                words_idx = [widx for widx in token_res[b_idx].words() if widx is not None]
                tokens_idx = [widx[0] for widx in enumerate(token_res[b_idx].words()) if widx[1] is not None]
                words_shape = list(enc.shape[1:])
                words_shape[1] = len(set(words_idx))
                enc_pooled = torch.zeros(tuple(words_shape)).to(self.device)
                enc_pooled.index_reduce_(1, torch.tensor(words_idx, dtype=torch.int64).to(self.device),
                                         enc[:, torch.tensor(tokens_idx, dtype=torch.int64).to(self.device)],
                                         "mean", include_self=False)
                enc[b_idx] = torch.squeeze(VECOPS[op](enc_pooled, dim=1))

        if (cache and cache_key not in PhraseEncoder._cache):
            PhraseEncoder._cache[cache_key] = enc

        return enc


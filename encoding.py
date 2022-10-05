import torch
import numpy as np
import gensim.downloader as gensim_api
from typing import List, Tuple
from torch import Tensor, device as Device
from transformers import T5EncoderModel, T5TokenizerFast
from transformers import BertModel, BertTokenizerFast
from transformers import RobertaModel, RobertaTokenizerFast
from sentence_transformers import SentenceTransformer
from data_access.adjectives_nouns import Adjectives, Nouns

VECOPS = {
    "sum": torch.sum,
    "mul": torch.prod,
    "mean": lambda x: torch.mean(x, dim=0)
}

ENCODERS = {
    "t5": ("sentence-transformers/sentence-t5-base", "t5-base", "transformer", T5EncoderModel, T5TokenizerFast),
    "bert": ("sentence-transformers/stsb-bert-base", "bert-base-cased", "transformer", BertModel, BertTokenizerFast),
    "distilroberta": ("sentence-transformers/all-distilroberta-v1", "distilroberta-base", "transformer", RobertaModel, RobertaTokenizerFast),
    "dpr": ("sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base", None, "transformer"),
    "labse": ("sentence-transformers/LaBSE", None, "transformer"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", None, "transformer"),
    "glove": ("glove-wiki-gigaword-300", None, "gensim"),
    "word2vec": ("word2vec-google-news-300", None, "gensim")
}


def get_transformer_model(enc_name: Tuple[str], sentence: bool, device: Device):
    tokenizer = None
    model = None
    if (sentence):
        model = SentenceTransformer(enc_name[0]).to(device)
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

    def encode(self, phrases: List[str], convert_to_tensor=True):
        dim = self._model["x"].shape
        embeddings = [np.mean([self._model[token] if (token in self._model) else np.zeros(dim) for token in phrase.split()], axis=0)
                      for phrase in phrases]

        return [torch.tensor(emb).to(self.device) for emb in embeddings]


class PhraseEncoder:
    def __init__(self, model_name: str = "t5", sentence: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._model = None
        self._sentence = sentence
        enc_name = ENCODERS[model_name]

        if (enc_name[2] == "transformer"):
            self._model, self._tokenizer = get_transformer_model(enc_name, sentence, self.device)
        elif (enc_name[2] == "gensim"):
            self._model = GensimModel(enc_name[0], self.device)
        else:
            raise EncoderNotSupportedException

    def encode(self, phrase: str, op: str = None) -> Tensor:
        if (self._sentence or type(self._model) == GensimModel):
            enc = self._model.encode([phrase], convert_to_tensor=True)[0]
        else:
            token_res = self._tokenizer(phrase, return_tensors="pt")
            input_ids = token_res.input_ids.to(self.device)
            enc = self._model(input_ids=input_ids).last_hidden_state
            words_idx = [widx for widx in token_res.words() if widx is not None]
            tokens_idx = [widx[0] for widx in enumerate(token_res.words()) if widx[1] is not None]
            words_shape = list(enc.shape)
            words_shape[1] = len(set(words_idx))
            enc_pooled = torch.zeros(tuple(words_shape)).to(self.device)
            enc_pooled.index_reduce_(1, torch.tensor(words_idx, dtype=torch.int64).to(self.device),
                                     enc[:, torch.tensor(tokens_idx, dtype=torch.int64).to(self.device)],
                                     "mean", include_self=False)
            enc = torch.squeeze(VECOPS[op](enc_pooled, dim=1))

        return enc


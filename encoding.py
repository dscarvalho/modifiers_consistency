import torch
from typing import List
from torch import Tensor, device as Device
from transformers import T5EncoderModel, T5TokenizerFast
from sentence_transformers import SentenceTransformer
from data_access.adjectives_nouns import Adjectives, Nouns

VECOPS = {
    "sum": torch.sum,
    "mul": torch.mul
}


def get_t5_model(sentence: bool, device: Device):
    tokenizer = None
    model = None
    if (sentence):
        model = SentenceTransformer("sentence-transformers/sentence-t5-base").to(device)
    else:
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        model = T5EncoderModel.from_pretrained("t5-small").to(device)

    return model, tokenizer


def phrase_from_index(idx_phrase: List[List[int]]):
    decoded = list()
    adjectives = Adjectives()
    nouns = Nouns()
    for idx_word, idx_type in idx_phrase:
        if (idx_type == 0):
            decoded.append(nouns.get_word(idx_word))
        else:
            decoded.append(adjectives.get_word(idx_word))

    return decoded


class EncoderNotSupportedException(Exception):
    def __init__(self, msg: str = "The encoder type is not supported."):
        self.message = msg


class PhraseEncoder:
    def __init__(self, model_name: str = "t5", sentence: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._model = None
        self._sentence = sentence

        if (model_name == "t5"):
            self._model, self._tokenizer = get_t5_model(sentence, self.device)
        else:
            raise EncoderNotSupportedException

    def encode(self, phrase: str, op: str = None) -> Tensor:
        if (self._sentence):
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
            enc = VECOPS[op](enc_pooled, dim=1)

        return enc


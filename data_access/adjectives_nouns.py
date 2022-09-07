from typing import Dict, Tuple
import pandas as pd


ADJ_PATH = "data/adjectives.csv"

ADJ_TYPES = {
    "Subsective (Intersective)": 1,
    "Subsective (Non-Intersective)": 2,
    "Plain Non-Subsective": 3,
    "Privative Non-Subsective": 4,
    "Ambiguous": 5
}

IDX_ADJ_TYPES = {v: k for k, v in ADJ_TYPES.items()}


class WordSet:
    def __init__(self):
        self._word_data: Dict[str, Tuple[int, int]] = dict()
        self._synonyms: Dict[int, str] = dict()
        self._index: Dict[int, str] = dict()

    def __getitem__(self, item: str):
        return self._word_data[item]

    def __iter__(self):
        for item in self._word_data:
            yield item, self._word_data[item]

    def __contains__(self, item):
        return item in self._word_data

    def __len__(self, item):
        return len(self._word_data)

    def get_word(self, idx: int) -> str:
        return self._index[idx]

    def get_synonym(self, word: int) -> str:
        return self._synonyms[word]


class Adjectives(WordSet):
    def __init__(self):
        super(Adjectives, self).__init__()
        adjectives_df = pd.read_csv(ADJ_PATH)
        idx = 0
        for adj_type in ADJ_TYPES:
            adjectives = adjectives_df[adj_type].tolist()
            for adj in adjectives:
                if (not pd.isna(adj)):
                    self._word_data[adj.lower()] = (idx, ADJ_TYPES[adj_type])
                    self._index[idx] = adj.lower()
                    idx += 1


class Nouns(WordSet):
    def __init__(self):
        super(Nouns, self).__init__()
        adjectives_df = pd.read_csv(ADJ_PATH)
        nouns = adjectives_df["Nouns"].tolist()
        synonyms = adjectives_df["Synonyms [Nouns]"].tolist()
        idx = 0
        for noun, synonym in zip(nouns, synonyms):
            if (not pd.isna(noun)):
                noun = noun.lower()
                self._word_data[noun] = (idx, 0)
                self._synonyms[idx] = synonym.lower()
                self._index[idx] = noun
                idx += 1

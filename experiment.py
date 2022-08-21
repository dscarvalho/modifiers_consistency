import json
import numpy as np
import torch
from typing import List, Dict, Tuple
from itertools import combinations
from dynaconf import settings
from encoding import PhraseEncoder, phrase_from_index
from data_access.adjectives_nouns import IDX_ADJ_TYPES


class SetDistanceExperiment:
    def __init__(self, filepath: str = settings["phrase_file_path"]):
        with open(filepath) as anfile:
            self.phrases: List[List[List[int]]] = json.load(anfile)
            self.group()

    def group(self):
        self.phrases.sort(key=lambda x: [(e[1], e[0]) for e in reversed(x)])

    def run(self) -> Dict[Tuple[int], float]:
        stats = dict()
        enc = PhraseEncoder("t5", True)
        i = 0
        for idx_phrase in [p for p in self.phrases if (len(p) > 1)]:
            # print("Phrase:", phrase_from_index(idx_phrase))
            phrase = " ".join(phrase_from_index(idx_phrase))
            vec = enc.encode(phrase, "sum")

            dist_phrase_word = [1 - torch.cosine_similarity(vec, enc.encode(" ".join(phrase_from_index([p])), "sum"), dim=0)
                                for p in idx_phrase]
            dist_comb_compl = list()
            for k in range(1, len(idx_phrase) + 1):
                combs = combinations(idx_phrase, k)
                dupl = set()
                for comb in combs:
                    if (tuple(phrase_from_index(comb)) not in dupl):
                        # print("Combination:", phrase_from_index(comb))
                        compl = [w for w in idx_phrase if w not in comb]
                        dupl.add(tuple(phrase_from_index(compl)))
                        if (compl):
                            # print("Complement:", phrase_from_index(compl))
                            dist_compl = 1 - torch.cosine_similarity(enc.encode(" ".join(phrase_from_index(comb)), "sum"),
                                                                     enc.encode(" ".join(phrase_from_index(compl)), "sum"),
                                                                     dim=0)
                            dist_comb_compl.append(dist_compl)

            if (tuple([w[1] for w in idx_phrase[:-1]]) not in stats):
                stats[tuple([w[1] for w in idx_phrase[:-1]])] = list()

            stats[tuple([w[1] for w in idx_phrase[:-1]])].append(np.mean([int((dpw <= dcc).item())
                                                                          for dpw in dist_phrase_word
                                                                          for dcc in dist_comb_compl]))
            if (i % 1000 == 0):
                print({tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats})
            i += 1

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


exp = SetDistanceExperiment()
exp.run()

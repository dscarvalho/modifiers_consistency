import sys
import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple
from itertools import combinations
from dynaconf import settings
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from encoding import PhraseEncoder, phrase_from_index, ENCODERS, VECOPS
from data_access.adjectives_nouns import Nouns, Adjectives, IDX_ADJ_TYPES


class Experiment:
    def __init__(self,
                 filepath: str = settings["phrase_file_path"],
                 encoder: str = settings["encoder"],
                 vecop: str = settings["vecop"],
                 sent_enc: bool = settings["sentence_encoding"],
                 strict: bool = settings["strict"]):
        with open(filepath) as anfile:
            self.phrases: List[List[List[int]]] = json.load(anfile)

        self.group()
        self.encoder: str = encoder
        self.vecop: str = vecop
        self.sent_enc: bool = sent_enc
        self.strict: bool = strict
        self._enc = PhraseEncoder(self.encoder, self.sent_enc)

    def group(self):
        self.phrases.sort(key=lambda x: [(e[1], e[0]) for e in reversed(x)])

    def run(self, noun: str = None):
        raise NotImplementedError


class SetDistanceExperiment(Experiment):

    def run(self, noun: str = None) -> Dict[Tuple[int], float]:
        stats = dict()
        nouns = Nouns()
        phrases = [p for p in self.phrases if (len(p) > 1 and (noun is None or list(nouns[noun]) in p))]
        i = 0
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            # print("Phrase:", phrase_from_index(idx_phrase))
            phrase = " ".join(phrase_from_index(idx_phrase))
            vec = self._enc.encode(phrase, self.vecop)

            dist_phrase_word = [1 - torch.cosine_similarity(vec, self._enc.encode(" ".join(phrase_from_index([p])), self.vecop), dim=0).item()
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
                            dist_compl = 1 - torch.cosine_similarity(self._enc.encode(" ".join(phrase_from_index(comb)), self.vecop),
                                                                     self._enc.encode(" ".join(phrase_from_index(compl)), self.vecop),
                                                                     dim=0).item()
                            dist_comb_compl.append(dist_compl)

            if (tuple([w[1] for w in idx_phrase[:-1]]) not in stats):
                stats[tuple([w[1] for w in idx_phrase[:-1]])] = list()

            if (self.strict):
                stats[tuple([w[1] for w in idx_phrase[:-1]])].append(int(False not in [dpw <= dcc
                                                                                       for dpw in dist_phrase_word
                                                                                       for dcc in dist_comb_compl]))
            else:
                stats[tuple([w[1] for w in idx_phrase[:-1]])].append([dpw <= dcc
                                                                      for dpw in dist_phrase_word
                                                                      for dcc in dist_comb_compl])
            if (i % 1000 == 0):
                print({tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats})
            i += 1

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


class AdjectiveTypeWeightExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[int, float]:
        stats_weight = {IDX_ADJ_TYPES[idx]: list() for idx in IDX_ADJ_TYPES}
        stats_weight_aggr = {IDX_ADJ_TYPES[idx]: 0.0 for idx in IDX_ADJ_TYPES}
        nouns = Nouns()
        phrases = [p for p in self.phrases if (len(p) > 1 and (noun is None or list(nouns[noun]) in p))]
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            phrase = " ".join(phrase_from_index(idx_phrase))
            vec = self._enc.encode(phrase, self.vecop)

            dist_phrase_word = [1 - torch.cosine_similarity(vec, self._enc.encode(" ".join(phrase_from_index([p])), self.vecop), dim=0).item()
                                for p in idx_phrase]

            for idx in IDX_ADJ_TYPES:
                stats_weight[IDX_ADJ_TYPES[idx]].append(sum([dist_phrase_word[i] for i in range(len(dist_phrase_word))
                                                         if idx_phrase[i][1] == idx]) / sum(dist_phrase_word))

        for idx in IDX_ADJ_TYPES:
            stats_weight_aggr[IDX_ADJ_TYPES[idx]] = np.mean(stats_weight[IDX_ADJ_TYPES[idx]])

        return stats_weight_aggr


class AdjectiveWeightExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[str, float]:
        adjectives = Adjectives()
        nouns = Nouns()
        stats_weight = {adj[0]: list() for adj in adjectives}
        stats_weight_aggr = {adj[0]: 0.0 for adj in adjectives}
        phrases = [p for p in self.phrases if (len(p) > 1 and (noun is None or list(nouns[noun]) in p))]
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            phrase = " ".join(phrase_from_index(idx_phrase))
            vec = self._enc.encode(phrase, self.vecop)

            dist_phrase_word = [1 - torch.cosine_similarity(vec, self._enc.encode(" ".join(phrase_from_index([p])), self.vecop), dim=0).item()
                                for p in idx_phrase]

            for adj in stats_weight:
                stats_weight[adj].append(sum([dist_phrase_word[i] for i in range(len(dist_phrase_word))
                                              if idx_phrase[i][0] == adjectives[adj][0]]) / sum(dist_phrase_word))

        for adj in stats_weight:
            stats_weight_aggr[adj] = np.mean(stats_weight[adj])

        return stats_weight_aggr


EXPERIMENTS = {
    "setdistance": SetDistanceExperiment,
    "adjtypeweight": AdjectiveTypeWeightExperiment,
    "adjweight": AdjectiveWeightExperiment
}


def exp_run(exp: Experiment):
    results = Parallel(n_jobs=cpu_count(True))(delayed(exp.run)(noun) for noun, idxs in Nouns())
    aggr = dict()
    avgs = dict()

    for res in results:
        for key in res:
            if (key not in aggr):
                aggr[key] = list()
            aggr[key].append(res[key])

    for key in aggr:
        avgs[key] = np.mean(aggr[key])

    exp_name = type(exp).__name__
    sent_config = "-sent" if exp.sent_enc else ""
    vecop = "-" + exp.vecop if not exp.sent_enc else ""
    strict = "-strict" if exp.strict else ""
    results_path = f"{settings['results_path']}/{exp_name}"

    if (not os.path.exists(results_path)):
        os.makedirs(results_path)

    with open(f"{results_path}/results_{exp.encoder}{sent_config}{vecop}{strict}.json", "w") as outfile:
        json.dump({str(key): avgs[key] for key in avgs}, outfile, indent=2)


def main(argv):
    if ("all" in argv):
        print("Running all...")
        for enc in ENCODERS:
            for vecop in VECOPS:
                for sent_enc in (True, False):
                    for strict in (True, False):
                        exp = SetDistanceExperiment(encoder=enc, vecop=vecop, sent_enc=sent_enc, strict=strict)
                        exp_run(exp)
    elif (len(argv) > 1 and argv[1] in EXPERIMENTS):
        exp = EXPERIMENTS[argv[1]]()
        sent_config = "sent" if exp.sent_enc else ""
        strict = "strict" if exp.strict else ""
        print("Running config:", settings["encoder"], settings["vecop"], sent_config, strict)
        exp_run(exp)
    else:
        print("Experiment not recognized. Must be one of: " + " | ".join(EXPERIMENTS.keys()))


if __name__ == "__main__":
    main(sys.argv)

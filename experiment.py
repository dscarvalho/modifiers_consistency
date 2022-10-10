import sys
import os
import json
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple
from itertools import combinations, product
from dynaconf import settings
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from encoding import PhraseEncoder, phrase_from_index, ENCODERS, VECOPS
from data_access.adjectives_nouns import Nouns, Adjectives, IDX_ADJ_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class Experiment:
    def __init__(self,
                 filepath: str = settings["phrase_file_path"],
                 encoder: str = settings["encoder"],
                 vecop: str = settings["vecop"],
                 sent_enc: bool = settings["sentence_encoding"],
                 strict: bool = settings["strict"],
                 control: int = settings["control"]):
        with open(filepath) as anfile:
            self.phrases: List[List[List[int]]] = json.load(anfile)

        self.group()
        self.encoder: str = encoder
        self.vecop: str = vecop
        self.sent_enc: bool = sent_enc
        self.strict: bool = strict
        self.control: int = control
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
        ctrl = self.control
        i = 0
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            # print("Phrase:", phrase_from_index(idx_phrase))
            phrase = " ".join(phrase_from_index(idx_phrase, ctrl < 0))
            vec = self._enc.encode(phrase, self.vecop)

            dist_phrase_word = [1 - torch.cosine_similarity(vec, self._enc.encode(" ".join(phrase_from_index([p], bool(ctrl & 1))), self.vecop),
                                                            dim=0).item()
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
                            dist_compl = 1 - torch.cosine_similarity(self._enc.encode(" ".join(phrase_from_index(comb, bool(ctrl & 2))), self.vecop),
                                                                     self._enc.encode(" ".join(phrase_from_index(compl, bool(ctrl & 2))), self.vecop),
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
            # if (i % 1000 == 0):
            #     print({tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats})
            i += 1

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


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

            for adj_dpw in [dpw for dpw in zip(idx_phrase, dist_phrase_word) if (dpw[0][1] > 0)]:
                stats_weight[adjectives.get_word(adj_dpw[0][0])].append(adj_dpw[1] / sum(dist_phrase_word))

        for adj in stats_weight:
            stats_weight_aggr[adj] = np.mean(stats_weight[adj])

        return stats_weight_aggr


class AdjectiveTypeWeightExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[int, float]:
        adjectives = Adjectives()
        stats_weight = {IDX_ADJ_TYPES[idx]: list() for idx in IDX_ADJ_TYPES}
        stats_weight_aggr = {IDX_ADJ_TYPES[idx]: 0.0 for idx in IDX_ADJ_TYPES}
        adj_stats = AdjectiveWeightExperiment.run(self, noun)

        for adj in adj_stats:
            stats_weight[IDX_ADJ_TYPES[adjectives[adj][1]]].append(adj_stats[adj])

        for idx in stats_weight:
            stats_weight_aggr[idx] = np.mean(stats_weight[idx])

        return stats_weight_aggr


class SynPhraseDistanceExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[Tuple[int], float]:
        stats = dict()
        nouns = Nouns()
        phrases = [p for p in self.phrases if (len(p) > 1 and (noun is None or list(nouns[noun]) in p))]
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            types = tuple([w[1] for w in idx_phrase[:-1]])
            phrase = " ".join(phrase_from_index(idx_phrase))
            syn_phrase = " ".join(phrase_from_index(idx_phrase, True))
            vec = self._enc.encode(phrase, self.vecop)
            syn_vec = self._enc.encode(syn_phrase, self.vecop)
            dist_syn_phrase = 1 - torch.cosine_similarity(vec, syn_vec, dim=0).item()

            if (types not in stats):
                stats[types] = list()

            stats[types].append(dist_syn_phrase)

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


class NonIntersectivityExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[Tuple[int], float]:
        stats = dict()
        nouns = Nouns()
        phrases = [p for p in self.phrases if (len(p) == 2 and (list(nouns[noun]) in p))]
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            if (self.control != 0):
                syn_cases = [(1, 0)] #product([0, 1], [0, 1])
            else:
                syn_cases = [(0, 0)]

            for adj_syn, noun_syn in syn_cases:
                types = tuple([w[1] for w in idx_phrase[:-1]])
                phrase = " ".join(phrase_from_index(idx_phrase))
                vec = self._enc.encode(phrase, self.vecop)

                dist_phrase_adj = 1 - torch.cosine_similarity(vec,
                                                              self._enc.encode(" ".join(phrase_from_index([idx_phrase[0]],
                                                                                                          bool(adj_syn))),
                                                                               self.vecop),
                                                              dim=0).item()
                dist_phrase_noun = 1 - torch.cosine_similarity(vec,
                                                               self._enc.encode(" ".join(phrase_from_index([idx_phrase[1]],
                                                                                                           bool(noun_syn))),
                                                                                self.vecop),
                                                               dim=0).item()

                if (types not in stats):
                    stats[types] = list()

                stats[types].append(int(dist_phrase_adj < dist_phrase_noun))

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


class PairSetDistanceExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[Tuple[int], float]:
        stats = dict()
        nouns = Nouns()
        adjectives = Adjectives()
        other_nouns = [nm[0] for nm in nouns if (nm[1][0] > nouns[noun][0])]
        adj_comb = set()

        for other_noun in tqdm(other_nouns, desc=f"Progress ({noun})"):
            for adj1 in adjectives:
                for adj2 in (adj for adj in adjectives if adj[0] != adj1[0]):
                    if ((adj1[1][0], adj2[1][0]) not in adj_comb):
                        adj_comb.add((adj1[1][0], adj2[1][0]))
                        adj_comb.add((adj2[1][0], adj1[1][0]))
                    else:
                        continue
                    types = (adj1[1][1], adj2[1][1])
                    phrases = (" ".join([adj1[0], noun]),
                               " ".join([adj1[0], other_noun]),
                               " ".join([adj2[0], noun]),
                               " ".join([adj2[0], other_noun]))
                    # print(list(phrases))
                    vecs = [self._enc.encode(phrase, self.vecop) for phrase in phrases]
                    dist_adj1 = 1 - torch.cosine_similarity(vecs[0], vecs[1], dim=0).item()
                    dist_adj2 = 1 - torch.cosine_similarity(vecs[2], vecs[3], dim=0).item()

                    if (types not in stats):
                        stats[types] = list()

                    stats[types].append(int(dist_adj1 < dist_adj2))

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


EXPERIMENTS = {
    "setdistance": SetDistanceExperiment,
    "adjtypeweight": AdjectiveTypeWeightExperiment,
    "adjweight": AdjectiveWeightExperiment,
    "synphrasedist": SynPhraseDistanceExperiment,
    "nonintersect": NonIntersectivityExperiment,
    "pairsetdist": PairSetDistanceExperiment
}


def exp_run(exp: Experiment):
    sent_config = "sent" if exp.sent_enc else ""
    strict = "strict" if exp.strict else ""
    control = f"-ctrl{exp.control}" if (exp.control and type(exp) == SetDistanceExperiment) else ""
    logger.info(f"Running {type(exp).__name__}")
    logger.info(f"Running config: {exp.encoder}, {exp.vecop}, {sent_config}, {strict}, {control}")

    results = Parallel(n_jobs=cpu_count(True) // 4)(delayed(exp.run)(noun) for noun, idxs in Nouns())
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

    with open(f"{results_path}/results_{exp.encoder}{sent_config}{vecop}{strict}{control}.json", "w") as outfile:
        json.dump({str(key): avgs[key] for key in avgs}, outfile, indent=2)


def main(argv):
    if ("all" in argv):
        logger.info("Running all...")
        for enc in ENCODERS:
                exp = SetDistanceExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
                exp = AdjectiveTypeWeightExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
                exp = SynPhraseDistanceExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
                exp = NonIntersectivityExperiment(encoder=enc, vecop="mean", sent_enc=True, strict=True)
                exp_run(exp)
    elif (len(argv) > 1 and argv[1] in EXPERIMENTS):
        exp = EXPERIMENTS[argv[1]]()
        exp_run(exp)
    else:
        logger.error("Experiment not recognized. Must be one of: " + " | ".join(EXPERIMENTS.keys()))


if __name__ == "__main__":
    main(sys.argv)

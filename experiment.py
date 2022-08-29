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
from data_access.adjectives_nouns import Nouns, IDX_ADJ_TYPES


class SetDistanceExperiment:
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

    def group(self):
        self.phrases.sort(key=lambda x: [(e[1], e[0]) for e in reversed(x)])

    def run(self, noun: str = None) -> Tuple[Dict[Tuple[int], float], Dict[int, float]]:
        stats = dict()
        stats_weight = {IDX_ADJ_TYPES[idx]: 0.0 for idx in IDX_ADJ_TYPES}
        enc = PhraseEncoder(self.encoder, self.sent_enc)
        nouns = Nouns()
        phrases = [p for p in self.phrases if (len(p) > 1 and (noun is None or list(nouns[noun]) in p))]
        i = 0
        for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
            # print("Phrase:", phrase_from_index(idx_phrase))
            phrase = " ".join(phrase_from_index(idx_phrase))
            vec = enc.encode(phrase, self.vecop)

            dist_phrase_word = [1 - torch.cosine_similarity(vec, enc.encode(" ".join(phrase_from_index([p])), self.vecop), dim=0).item()
                                for p in idx_phrase]

            for idx in IDX_ADJ_TYPES:
                stats_weight[IDX_ADJ_TYPES[idx]] += sum([dist_phrase_word[i] for i in range(len(dist_phrase_word))
                                                         if idx_phrase[i][1] == idx]) / sum(dist_phrase_word)
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
                            dist_compl = 1 - torch.cosine_similarity(enc.encode(" ".join(phrase_from_index(comb)), self.vecop),
                                                                     enc.encode(" ".join(phrase_from_index(compl)), self.vecop),
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

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}, stats_weight


def exp_run(exp: SetDistanceExperiment):
    results = Parallel(n_jobs=cpu_count(True))(delayed(exp.run)(noun) for noun, idxs in Nouns())
    totals = dict()
    totals_weights = {IDX_ADJ_TYPES[idx]: 0.0 for idx in IDX_ADJ_TYPES}

    for res, weights in results:
        for key in res:
            if (key not in totals):
                totals[key] = 0
            totals[key] += res[key]

        for key in weights:
            totals_weights[key] += weights[key]

    for key in totals:
        totals[key] /= len(results)

    for key in totals_weights:
        totals_weights[key] /= len(exp.phrases)

    sent_config = "-sent" if exp.sent_enc else ""
    vecop = "-" + exp.vecop if not exp.sent_enc else ""
    strict = "-strict" if exp.strict else ""

    if (not os.path.exists(settings["results_path"])):
        os.makedirs(settings["results_path"])

    with open(f"{settings['results_path']}/results_{exp.encoder}{sent_config}{vecop}{strict}.json", "w") as outfile:
        json.dump({str(key): totals[key] for key in totals}, outfile, indent=2)

    with open(f"{settings['results_path']}/weights_{exp.encoder}{sent_config}{vecop}{strict}.json", "w") as outfile:
        json.dump(totals_weights, outfile, indent=2)


def main(argv):
    if ("all" in argv):
        print("Running all...")
        for enc in ENCODERS:
            for vecop in VECOPS:
                for sent_enc in (True, False):
                    for strict in (True, False):
                        exp = SetDistanceExperiment(encoder=enc, vecop=vecop, sent_enc=sent_enc, strict=strict)
                        exp_run(exp)
    else:
        exp = SetDistanceExperiment()
        sent_config = "sent" if exp.sent_enc else ""
        vecop = "-" + exp.vecop if not exp.sent_enc else ""
        strict = "strict" if exp.strict else ""
        print("Running config:", settings["encoder"], settings["vecop"], sent_config, strict)
        exp_run(exp)



if __name__ == "__main__":
    main(sys.argv)

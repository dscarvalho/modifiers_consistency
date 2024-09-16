import os
import json
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple
from itertools import combinations
from dynaconf import settings
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from encoding import PhraseEncoder, phrase_from_index
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
                 control: int = settings["control"],
                 batch_size: int = settings["batch_size"]):
        with open(filepath) as anfile:
            self.phrases: List[List[List[int]]] = json.load(anfile)

        self.group()
        self.encoder: str = encoder
        self.vecop: str = vecop
        self.sent_enc: bool = sent_enc
        self.strict: bool = strict
        self.control: int = control
        self.batch_size: int = batch_size
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
        bsize = self.batch_size
        for i in tqdm(range(0, len(phrases), bsize), desc=f"Progress ({noun})"):
            # print("Phrase:", phrase_from_index(idx_phrase))
            batch_phrases = [" ".join(phrase_from_index(idxs, ctrl < 0)) for idxs in phrases[i:i + bsize]]
            vec_phrases = self._enc.encode(batch_phrases, self.vecop)
            max_words = max([len(p) for p in phrases[i:i + bsize]])

            vec_words = torch.zeros(vec_phrases.shape[0], max_words, vec_phrases.shape[-1], device=self._enc.device)
            for j in range(vec_phrases.shape[0]):
                for k in range(len(phrases[i:i + bsize][j])):
                    vec_words[j, k] = self._enc.encode(
                        [" ".join(phrase_from_index([phrases[i:i + bsize][j][k]], bool(ctrl & 1)))],
                        self.vecop,
                        cache=True
                    )

            dist_phrases_words = torch.stack([1 - torch.cosine_similarity(vec_phrases, vec_words[:, k], dim=-1)
                                              for k in range(vec_words.shape[1])]).T


            # dist_phrases_words_ = list()
            # for j in range(vec_phrases.shape[0]):
            #     dist_phrase_word = [
            #         1 - torch.cosine_similarity(
            #             vec_phrases[j], self._enc.encode([" ".join(phrase_from_index([p], bool(ctrl & 1)))], self.vecop, cache=True),
            #             dim=-1
            #         ).item()
            #         for p in phrases[i:i + bsize][j]
            #     ]
            #     dist_phrases_words_.append(dist_phrase_word)


            compl_phrases_idxs = dict()
            max_combs = 0
            # dist_comb_compl_ = list()
            for j in range(vec_phrases.shape[0]):
                for k in range(1, len(phrases[i:i + bsize][j]) + 1):
                    combs = combinations(phrases[i:i + bsize][j], k)
                    dupl = set()
                    for c, comb in enumerate(combs):
                        if (tuple(phrase_from_index(comb)) not in dupl):
                            # print("Combination:", phrase_from_index(comb))
                            compl = [w for w in phrases[i:i + bsize][j] if w not in comb]
                            dupl.add(tuple(phrase_from_index(compl)))
                            if (compl):
                                comb_idx = (k - 1) * len(phrases[i:i + bsize][j]) + c
                                max_combs = max(comb_idx + 1, max_combs)
                                # print("Complement:", phrase_from_index(compl))
                                compl_pair = (" ".join(phrase_from_index(comb, bool(ctrl & 2))),
                                              " ".join(phrase_from_index(compl, bool(ctrl & 2))))
                                for p_idx, p in enumerate(compl_pair):
                                    if (p not in compl_phrases_idxs):
                                        compl_phrases_idxs[p] = list()
                                    compl_phrases_idxs[p].append((j, comb_idx, p_idx))

                                # dist_compl = 1 - torch.cosine_similarity(self._enc.encode(" ".join(phrase_from_index(comb, bool(ctrl & 2))), self.vecop),
                                #                                          self._enc.encode(" ".join(phrase_from_index(compl, bool(ctrl & 2))), self.vecop),
                                #                                          dim=0).item()
                                # dist_comb_compl_.append(dist_compl)
            compl_phrases = list(compl_phrases_idxs.keys())
            compl_embs = self._enc.encode(compl_phrases, self.vecop)
            vec_compls = torch.zeros(vec_phrases.shape[0], max_combs, 2, vec_phrases.shape[-1], device=self._enc.device)
            for compl_idx in range(len(compl_phrases)):
                emb = compl_embs[compl_idx]
                for idxs in compl_phrases_idxs[compl_phrases[compl_idx]]:
                    vec_compls[*idxs] = emb

            dist_comb_compl = 1 - torch.cosine_similarity(vec_compls[:, :, 0], vec_compls[:, :, 1], dim=-1)

            for j in range(vec_phrases.shape[0]):
                type_conf = tuple([w[1] for w in phrases[i:i + bsize][j][:-1]])
                if (type_conf not in stats):
                    stats[type_conf] = list()

                # if (self.strict):
                intersec_test = int(False not in set([(dpw <= dcc).cpu().item()
                                                      for dpw in dist_phrases_words[j]
                                                      for dcc in dist_comb_compl[j]
                                                      if (dpw.item() < 1 and dcc.item() < 1)]))  # Suppresses padding
                stats[type_conf].append(intersec_test)
                # else:
                #     stats[tuple([w[1] for w in phrases[i:i + bsize][j][:-1]])].append([dpw <= dcc
                #                                                                        for dpw in dist_phrase_word
                #                                                                        for dcc in dist_comb_compl])
                # # if (i % 1000 == 0):
                # #     print({tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats})

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


class AdjectiveWeightExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[str, float]:
        adjectives = Adjectives()
        nouns = Nouns()
        stats_weight = {adj[0]: list() for adj in adjectives}
        stats_weight_aggr = {adj[0]: 0.0 for adj in adjectives}
        phrases = [p for p in self.phrases if (len(p) > 1 and (noun is None or list(nouns[noun]) in p))]
        bsize = self.batch_size

        for i in tqdm(range(0, len(phrases), bsize), desc=f"Progress ({noun})"):
            batch_phrases = [" ".join(phrase_from_index(idxs)) for idxs in phrases[i:i + bsize]]
            vec_phrases = self._enc.encode(batch_phrases, self.vecop)
            max_words = max([len(p) for p in phrases[i:i + bsize]])

            vec_words = torch.zeros(vec_phrases.shape[0], max_words, vec_phrases.shape[-1], device=self._enc.device)
            for j in range(vec_phrases.shape[0]):
                for k in range(len(phrases[i:i + bsize][j])):
                    vec_words[j, k] = self._enc.encode(
                        [" ".join(phrase_from_index([phrases[i:i + bsize][j][k]]))],
                        self.vecop,
                        cache=True
                    )

            dist_phrases_words = torch.stack([1 - torch.cosine_similarity(vec_phrases, vec_words[:, k], dim=-1)
                                              for k in range(vec_words.shape[1])]).T

            for j in range(vec_phrases.shape[0]):
                for adj_dpw in [dpw for dpw in zip(phrases[i:i + bsize][j], dist_phrases_words[j]) if (dpw[0][1] > 0)]:
                    stats_weight[adjectives.get_word(adj_dpw[0][0])].append(adj_dpw[1].item() / dist_phrases_words[j].sum().item())

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
        bsize = self.batch_size

        for i in tqdm(range(0, len(phrases), bsize), desc=f"Progress ({noun})"):
            batch_phrases = list()
            syn_phrases = list()
            phrase_types = list()
            for idx_phrase in phrases[i:i + bsize]:
                types = tuple([w[1] for w in idx_phrase[:-1]])
                phrase = " ".join(phrase_from_index(idx_phrase))
                syn_phrase = " ".join(phrase_from_index(idx_phrase, True))
                batch_phrases.append(phrase)
                syn_phrases.append(syn_phrase)
                phrase_types.append(types)
                if (types not in stats):
                    stats[types] = list()

            vec_phrases = self._enc.encode(batch_phrases, self.vecop)
            vec_syn_phrases = self._enc.encode(syn_phrases, self.vecop)
            dist_syn_phrases = 1 - torch.cosine_similarity(vec_phrases, vec_syn_phrases, dim=-1)

            for j in range(vec_phrases.shape[0]):
                stats[phrase_types[j]].append(dist_syn_phrases[j].item())

        # for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
        #     types = tuple([w[1] for w in idx_phrase[:-1]])
        #     phrase = " ".join(phrase_from_index(idx_phrase))
        #     syn_phrase = " ".join(phrase_from_index(idx_phrase, True))
        #     vec = self._enc.encode(phrase, self.vecop)
        #     syn_vec = self._enc.encode(syn_phrase, self.vecop)
        #     dist_syn_phrase = 1 - torch.cosine_similarity(vec, syn_vec, dim=0).item()
        #
        #     if (types not in stats):
        #         stats[types] = list()
        #
        #     stats[types].append(dist_syn_phrase)

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


class NonSubsectivityExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[Tuple[int], float]:
        stats = dict()
        nouns = Nouns()
        phrases = [p for p in self.phrases if (len(p) == 2 and (list(nouns[noun]) in p))]
        bsize = self.batch_size
        if (self.control != 0):
            syn_cases = [(1, 0)]  # product([0, 1], [0, 1])
        else:
            syn_cases = [(0, 0)]

        for i in tqdm(range(0, len(phrases), bsize), desc=f"Progress ({noun})"):
            batch_phrases = list()
            phrase_types = list()
            adj_syn_batch_phrases = list()
            noun_syn_batch_phrases = list()
            for idx_phrase in phrases[i:i + bsize]:
                types = tuple([w[1] for w in idx_phrase[:-1]])
                phrase = " ".join(phrase_from_index(idx_phrase))
                batch_phrases.append(phrase)
                phrase_types.append(types)
                if (types not in stats):
                    stats[types] = list()

                for adj_syn, noun_syn in syn_cases:
                    adj_syn_batch_phrases.append(" ".join(phrase_from_index([idx_phrase[0]], bool(adj_syn))))
                    noun_syn_batch_phrases.append(" ".join(phrase_from_index([idx_phrase[1]], bool(noun_syn))))

            vec_phrases = self._enc.encode(batch_phrases, self.vecop)
            vec_adj_syn_phrases = self._enc.encode(adj_syn_batch_phrases, self.vecop)
            vec_noun_syn_phrases = self._enc.encode(noun_syn_batch_phrases, self.vecop)

            dist_phrases_adj = 1 - torch.cosine_similarity(vec_phrases, vec_adj_syn_phrases, dim=-1)
            dist_phrases_noun = 1 - torch.cosine_similarity(vec_phrases, vec_noun_syn_phrases, dim=-1)
            dist_adj_noun_comp = (dist_phrases_adj < dist_phrases_noun).to(torch.int8)

            for j in range(vec_phrases.shape[0]):
                stats[phrase_types[j]].append(dist_adj_noun_comp[j].item())

        # for idx_phrase in tqdm(phrases, desc=f"Progress ({noun})"):
        #     if (self.control != 0):
        #         syn_cases = [(1, 0)] #product([0, 1], [0, 1])
        #     else:
        #         syn_cases = [(0, 0)]
        #
        #     for adj_syn, noun_syn in syn_cases:
        #         types = tuple([w[1] for w in idx_phrase[:-1]])
        #         phrase = " ".join(phrase_from_index(idx_phrase))
        #         vec = self._enc.encode(phrase, self.vecop)
        #
        #         dist_phrase_adj = 1 - torch.cosine_similarity(vec,
        #                                                       self._enc.encode(" ".join(phrase_from_index([idx_phrase[0]],
        #                                                                                                   bool(adj_syn))),
        #                                                                        self.vecop),
        #                                                       dim=0).item()
        #         dist_phrase_noun = 1 - torch.cosine_similarity(vec,
        #                                                        self._enc.encode(" ".join(phrase_from_index([idx_phrase[1]],
        #                                                                                                    bool(noun_syn))),
        #                                                                         self.vecop),
        #                                                        dim=0).item()
        #
        #         if (types not in stats):
        #             stats[types] = list()
        #
        #         stats[types].append(int(dist_phrase_adj < dist_phrase_noun))

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


class PairSetDistanceExperiment(Experiment):
    def run(self, noun: str = None) -> Dict[Tuple[int], float]:
        stats = dict()
        nouns = Nouns()
        adjectives = Adjectives()
        other_nouns = [nm[0] for nm in nouns if (nm[1][0] > nouns[noun][0])]
        adj_comb = set()
        phrases_idx = dict()
        batch_phrases = list()
        phrase_types = list()

        for other_noun in tqdm(other_nouns, desc=f"Progress ({noun})"):
            for adj1 in adjectives:
                for adj2 in (adj for adj in adjectives if adj[0] != adj1[0]):
                    types = (adj1[1][1], adj2[1][1])
                    phrases = (" ".join([adj1[0], noun]),
                               " ".join([adj1[0], other_noun]),
                               " ".join([adj2[0], noun]),
                               " ".join([adj2[0], other_noun]))
                    # print(list(phrases))
                    for phrase in phrases:
                        if (phrase not in phrases_idx):
                            phrases_idx[phrase] = len(phrases_idx)

                    batch_phrases.append(tuple([phrases_idx[phrase] for phrase in phrases]))
                    phrase_types.append(types)
                    # vecs = [self._enc.encode(phrase, self.vecop) for phrase in phrases]
                    # dist_adj1 = 1 - torch.cosine_similarity(vecs[0], vecs[1], dim=0).item()
                    # dist_adj2 = 1 - torch.cosine_similarity(vecs[2], vecs[3], dim=0).item()

                    if (types not in stats):
                        stats[types] = list()

        phrases = list(sorted(phrases_idx.keys(), key=lambda x: phrases_idx[x]))
        if (phrases):
            vec_phrases = self._enc.encode(phrases, self.vecop)
            for i in tqdm(range(len(batch_phrases)), desc="Dist. calculation"):
                vecs = [vec_phrases[idx] for idx in batch_phrases[i]]
                dist_adj1 = 1 - torch.cosine_similarity(vecs[0], vecs[1], dim=0).item()
                dist_adj2 = 1 - torch.cosine_similarity(vecs[2], vecs[3], dim=0).item()
                stats[phrase_types[i]].append(int(dist_adj1 < dist_adj2))


        # for other_noun in tqdm(other_nouns, desc=f"Progress ({noun})"):
        #     for adj1 in adjectives:
        #         for adj2 in (adj for adj in adjectives if adj[0] != adj1[0]):
        #             # if ((adj1[1][0], adj2[1][0]) not in adj_comb):
        #             #     adj_comb.add((adj1[1][0], adj2[1][0]))
        #             #     adj_comb.add((adj2[1][0], adj1[1][0]))
        #             # else:
        #             #     continue
        #             types = (adj1[1][1], adj2[1][1])
        #             phrases = (" ".join([adj1[0], noun]),
        #                        " ".join([adj1[0], other_noun]),
        #                        " ".join([adj2[0], noun]),
        #                        " ".join([adj2[0], other_noun]))
        #             # print(list(phrases))
        #             vecs = [self._enc.encode(phrase, self.vecop) for phrase in phrases]
        #             dist_adj1 = 1 - torch.cosine_similarity(vecs[0], vecs[1], dim=0).item()
        #             dist_adj2 = 1 - torch.cosine_similarity(vecs[2], vecs[3], dim=0).item()
        #
        #             if (types not in stats):
        #                 stats[types] = list()
        #
        #             stats[types].append(int(dist_adj1 < dist_adj2))

        return {tuple([IDX_ADJ_TYPES[t] for t in k]): np.mean(stats[k]) for k in stats}


EXPERIMENTS = {
    "setdistance": SetDistanceExperiment,
    "adjtypeweight": AdjectiveTypeWeightExperiment,
    "adjweight": AdjectiveWeightExperiment,
    "synphrasedist": SynPhraseDistanceExperiment,
    "nonintersect": NonSubsectivityExperiment,
    "pairsetdist": PairSetDistanceExperiment
}


def exp_run(exp: Experiment):
    sent_config = "sent" if exp.sent_enc else ""
    strict = "strict" if exp.strict else ""
    control = f"-ctrl{exp.control}" if (exp.control and type(exp) == SetDistanceExperiment) else ""
    logger.info(f"Running {type(exp).__name__}")
    logger.info(f"Running config: {exp.encoder}, {exp.vecop}, {sent_config}, {strict}, {control}")

    if ("text-embeddings-" in exp.encoder or "nv-embed" in exp.encoder or "stella" in exp.encoder):
        results = [exp.run(noun) for noun, idxs in Nouns()]
    else:
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



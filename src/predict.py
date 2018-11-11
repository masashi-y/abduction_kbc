
import argparse
import dill
import sys
import numpy as np

from processors.evaluator import Evaluator
from utils.dataset import TripletDataset, Vocab
from utils.graph import *
from models.complex import ComplEx

def cal_rank(score_mat):
    return np.argsort(-score_mat, axis=1)

def run(model, dataset, ent_vocab, rel_vocab,
            lemma2id, id2lemma, batchsize=100, topn=100):
    pairs = [(sid, rid, oid) for s, o in dataset
                for sid in lemma2id[s]
                    for oid in lemma2id[o]
                        for rid in range(len(rel_vocab))]

    if len(pairs) == 0:
        print("NO ENTRIES FOR:", dataset, file=sys.stderr)
        return
    samples = np.asarray(pairs)
    res = model.cal_triplet_scores(samples)
    rank = np.argsort(-res)

    for i, (score, (sid, rid, oid)) in enumerate(
                                zip(res[rank], samples[rank]), 1):
        print("{: >3}: {}\t{}\t{}\t{:.3f}".format(
            i,
            # id2lemma[sid],
            # id2lemma[oid],
            ent_vocab.id2word[sid],
            ent_vocab.id2word[oid],
            rel_vocab.id2word[rid],
            score))
        if i >= topn:
            break
    print()

# def run(model, dataset, batchsize=100):
#     res_rank = []
#     res_score = []
#     for samples in dataset.batch_iter(batchsize, rand_flg=False):
#         subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
#         scores = model.cal_scores(subs, rels)
#         ranks = cal_rank(scores)
#         idx = np.arange(len(scores))[:, np.newaxis]
#         res_rank.append(ranks)
#         res_score.append(scores[idx, ranks])
#     return np.concatenate(res_rank), np.concatenate(res_score)
#
def clean_ent(ent):
    ent = ent[2:]
    items = ent.split("_")
    syn_id = items[-1]
    cat = items[-2]
    ent = "_".join(items[:-2])
    return ent

def compress(args):
    from collections import defaultdict
    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)
    model = ComplEx.load_model(args.model)
    lemma2id = defaultdict(list)
    id2lemma = []
    for i, w in enumerate(ent_vocab.id2word):
        # w = clean_ent(w)
        w = w.split(".")[0]
        lemma2id[w].append(i)
        id2lemma.append(w)
    with open(args.out, "wb") as f:
        dill.dump([ent_vocab, rel_vocab, model, lemma2id, id2lemma], f)

def prompt():
    def parse(pair, bracket=True):
        if bracket:
            assert pair[0] == "(" and pair[-1] == ")"
            pair = pair[1:-1]
        s, o = pair.split(",")
        return s.strip(), o.strip()
    res = []
    input_line = input(">> ").strip()
    inp = input_line
    try:
        while len(inp) > 0:
            idx = inp.find(")")
            if idx == -1:
                res.append(parse(inp, bracket=False))
                break
            else:
                res.append(parse(inp[:idx+1]))
                inp = inp[idx+1:].strip(", ")
    except:
        print(("PARSER ERROR ON INPUT : {}\n"
              "input line should be either of:\n"
              "  * HEAD, TAIL\n"
              "  * (HEAD, TAIL) [(HEAD, TAIL) ...]\n"
              "  * (HEAD, TAIL), [(HEAD, TAIL) ...]\n"
              ).format(input_line), file=sys.stderr)
        return None
    return res

def test(args):
    with open(args.model, "rb") as f:
        ent_vocab, rel_vocab, model, lemma2id, id2lemma = dill.load(f)

    def get_ent(i):
        return clean_ent(ent_vocab.id2word[i])

    while True:
        try:
            dataset = prompt()
            if dataset is not None and len(dataset) > 0:
                run(model, dataset, ent_vocab,
                        rel_vocab, lemma2id, id2lemma,
                        batchsize=100, topn=args.topn)
        except KeyboardInterrupt:
            print()



if __name__ == '__main__':
    p = argparse.ArgumentParser('Link prediction models')
    p.set_defaults(func=lambda _: p.print_help())
    ps = p.add_subparsers()

    p1 = ps.add_parser("create")
    p1.add_argument('--ent', type=str, help='entity list')
    p1.add_argument('--rel', type=str, help='relation list')
    p1.add_argument('--model', type=str, help='trained model path')
    p1.add_argument('--out', type=str, help='output file', default="model.config")
    p1.set_defaults(func=compress)

    p2 = ps.add_parser("run")
    p2.add_argument('--model', type=str, help='trained model config file')
    p2.add_argument('--topn', type=int, help='show top N results', default=3)
    p2.set_defaults(func=test)

    args = p.parse_args()
    args.func(args)


from message_pb2 import *
import numpy as np
import os
import sys
import argparse
import socket
import struct
import chainer
import multiprocessing
import dill
from conve import ConvE, ComplEx, Vocab
from normalization import denormalize_token


models = {'complex': ComplEx, 'conve': ConvE}


def log(msg):
    print("[server]", msg, file=sys.stderr)


def send_msg(sock, msg):
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)


def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def daemonize():
    def fork():
        if os.fork():
            sys.exit()

    def throw_away_io():
        stdin = open(os.devnull, 'rb')
        stdout = open(os.devnull, 'ab+')
        stderr = open(os.devnull, 'ab+', 0)

        for (null_io, std_io) in zip((stdin, stdout, stderr),
                             (sys.stdin, sys.stdout, sys.stderr)):
            os.dup2(null_io.fileno(), std_io.fileno())
    fork()
    os.setsid()
    fork()
    throw_away_io()


def compress(args):
    from collections import defaultdict
    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)
    model = models[args.model](len(ent_vocab), len(rel_vocab))
    chainer.serializers.load_npz(args.modelfile, model)
    lemma2id = defaultdict(list)
    id2lemma = []
    for i, w in enumerate(ent_vocab.id2word):
        lemma2id[w].append(i)
        id2lemma.append(w)
    with open(args.out, "wb") as f:
        dill.dump([ent_vocab, rel_vocab, model, lemma2id, id2lemma], f)


def cal_rank(score_mat):
    return np.argsort(-score_mat, axis=1)


def make_rank_complex(rank_from_coq, ent_vocab, rel_vocab, model, lemma2id, threshold=0.9):
    res = []
    nrels = len(rel_vocab)
    cands = []
    seen = []
    id2pred = dict()
    for cand in rank_from_coq.list:
        concl_pred = cand.pred1
        prem_pred = cand.pred2

        concl_str = denormalize_token(concl_pred.str)
        prem_str = denormalize_token(prem_pred.str)
        concl_base = concl_str
        prem_base = prem_str

        cp = lemma2id.get(concl_base)
        pp = lemma2id.get(prem_base)
        if cp is not None and pp is not None \
                and cp != pp and (cp, pp) not in seen:
            cands.extend((sid, rid, oid) for sid in pp for oid in cp for rid in range(nrels))
            for sid in pp:
                id2pred[sid] = prem_pred
            for oid in cp:
                id2pred[oid] = concl_pred
            seen.append((cp, pp))

    if len(cands) > 0:
        samples = np.asarray(cands)
        e1, rel, e2 = map(np.asarray, zip(*cands))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            pred = model.forward(e1, rel, e2)
        pred = pred.data
        rank = np.argsort(-pred)
        for score, (sid, rid, oid) in zip(pred[rank], samples[rank]):
            if score < threshold:
                break
            s, o = id2pred[sid], id2pred[oid]
            r = rel_vocab.id2word[rid]
            print(s.str, r, o.str, score, file=sys.stderr)
            if r == 'antonyms':
                res.append(Candidate(pred1=o, pred2=s, rel='anto'))
            elif r == 'hyponyms':
                res.append(Candidate(pred1=o, pred2=s, rel='hypo'))
            elif r == 'derivationally-related':
                res.append(Candidate(pred1=o, pred2=s, rel='entail'))
            elif r == 'hypernyms':
                res.append(Candidate(pred1=o, pred2=s, rel='entail'))
            elif r == 'also_sees':
                res.append(Candidate(pred1=o, pred2=s, rel='entail'))
    return Rank(list=res)


def on_other_thread(sock, args):
    msg = recv_msg(sock)
    msg = Echo.FromString(msg)
    if msg.msg == "Connecting":
        res = Echo(msg="OK")
    else:
        log("received: {} items".format(len(msg.rank.list)))
        res = Echo(rank=make_rank_complex(msg.rank, *args))
    print(res)
    res = res.SerializeToString()
    log('sending')
    send_msg(sock, res)
    sock.close()
    log('done')


def run(args):
    with open(args.model, "rb") as f:
        ent_vocab, rel_vocab, model, lemma2id, _ = dill.load(f)
        model_args = (ent_vocab, rel_vocab, model, lemma2id, args.threshold)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.bind(args.filename)

            if args.daemon:
                daemonize()

            while True:
                log('listening')
                s.listen(5)
                c, addr = s.accept()
                log('receiving')
                multiprocessing.Process(target=on_other_thread, args=(c, model_args)).start()
    except KeyboardInterrupt:
        os.unlink(args.filename)


def main():
    parser = argparse.ArgumentParser(description="python server for abduction tactic")
    parser.set_defaults(func=lambda _: parser.print_help())
    ps = parser.add_subparsers()

    p1 = ps.add_parser("create")
    p1.add_argument('--ent', type=str, help='entity list')
    p1.add_argument('--rel', type=str, help='relation list')
    p1.add_argument('--model', type=str, choices=models.keys())
    p1.add_argument('--modelfile', type=str, help='trained model path')
    p1.add_argument('--out', type=str, help='output file', default="model.config")
    p1.set_defaults(func=compress)

    p2 = ps.add_parser("run")
    p2.add_argument("--filename", type=str, default="/tmp/py_server", help="unix domain socket")
    p2.add_argument("--daemon", action="store_true")
    p2.add_argument("--threshold", type=float, default=0.9, help="threshold")
    p2.add_argument("model", type=str, default="model.config", help="ComplEx model config file")
    p2.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

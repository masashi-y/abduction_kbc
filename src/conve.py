#!/usr/bin/env python
import argparse
from collections import defaultdict

import numpy as np
import os
from tqdm import tqdm
import logging
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class BaseModel(object):
    def binary_cross_entropy(self, probs, Y):
        """
        Input:
            probs: probability matrix in any shape
            Y {0, 1} matrix in the same shape as probs
        Output:
            scalar loss
        """
        losses = Y * F.log(probs + 1e-6) + (1 - Y) * F.log(1 - probs + 1e-6)
        loss = - F.average(losses)
        return loss

    def evaluate(self, probs, e2, flt):
        """
        Input:
            probs: probability matrix (shape: (batch_size, num_entities) )
            e2: gold e2's (shape: (batch_size,) )
            flt: {0,1} filters for filtered scores (shape: (batch_size, num_entities) )
        Output:
            MRR, filtered MRR, filtered HITs (1, 3, 10)
        """
        batch_size, = e2.shape
        rank_all = self.xp.argsort(-probs.data)
        probs_flt = probs * flt
        rank_all_flt = self.xp.argsort(-probs_flt.data)
        mrr = mrr_flt = hits1 = hits3 = hits10 = 0.
        for i in range(batch_size):
            rank = self.xp.where(rank_all[i] == e2[i])[0][0] + 1
            mrr += 1. / rank
            rank_flt = self.xp.where(rank_all_flt[i] == e2[i])[0][0] + 1
            mrr_flt += 1. / rank_flt
            if rank_flt <= 1:
                hits1 += 1
            if rank_flt <= 3:
                hits3 += 1
            if rank_flt <= 10:
                hits10 += 1
        mrr /= float(batch_size)
        mrr_flt /= float(batch_size)
        hits1 /= float(batch_size)
        hits3 /= float(batch_size)
        hits10 /= float(batch_size)
        return {'mrr': mrr,
                'mrr(flt)': mrr_flt,
                'hits1(flt)': hits1,
                'hits3(flt)': hits3,
                'hits10(flt)': hits10}

    def __call__(self, e1, rel, e2, Y, flt):
        """
        Input:
            e1, rel, e2: ids for each entity and relation (shape : (batchsize,) )
            Y: whether true (1) or negative (0) sample (shape : (batchsize,) )
            flt: {0, 1} array used for filter evaluation (shape : (batch_size, num_entities) )
        Output:
            loss ( float )
        """
        if chainer.config.train:
            probs = self.forward(e1, rel, e2)
            loss = self.binary_cross_entropy(probs, Y)
            reporter.report({'loss': loss}, self)
            return loss
        else:
            assert flt is not None
            batch_size, = e1.shape
            probs_all = self.forward(e1, rel, None)
            metrics = self.evaluate(probs_all, e2, flt)
            probs = probs_all[self.xp.arange(batch_size), e2]
            loss = self.binary_cross_entropy(probs, Y)
            metrics['loss'] = loss
            reporter.report(metrics, self)
            return loss


class ComplEx(chainer.Chain, BaseModel):
    """
    Complex Embeddings for Simple Link Prediction, Théo Trouillon et al., 2016
    """
    def __init__(self, num_entities, num_relations, embedding_dim=200):
        super(ComplEx, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        with self.init_scope():
            self.emb_e_real = L.EmbedID(
                    num_entities, embedding_dim, initialW=I.GlorotNormal())
            self.emb_e_img = L.EmbedID(
                    num_entities, embedding_dim, initialW=I.GlorotNormal())
            self.emb_rel_real = L.EmbedID(
                    num_relations, embedding_dim, initialW=I.GlorotNormal())
            self.emb_rel_img = L.EmbedID(
                    num_relations, embedding_dim, initialW=I.GlorotNormal())

    def forward(self, e1, rel, e2=None):
        """
        Input:
            e1, rel, e2: ids for each entity and relation (shape : (batchsize,) )
        Output:
            score (shape : (batchsize,) )
        """
        batch_size, = e1.shape
        e1_embedded_real = self.emb_e_real(e1).reshape(batch_size, -1)
        rel_embedded_real = self.emb_rel_real(rel).reshape(batch_size, -1)
        e1_embedded_img = self.emb_e_img(e1).reshape(batch_size, -1)
        rel_embedded_img = self.emb_rel_img(rel).reshape(batch_size, -1)

        e1_embedded_real = F.dropout(e1_embedded_real, 0.2)
        rel_embedded_real = F.dropout(rel_embedded_real, 0.2)
        e1_embedded_img = F.dropout(e1_embedded_img, 0.2)
        rel_embedded_img = F.dropout(rel_embedded_img, 0.2)

        if e2 is not None:
            e2_embedded_real = self.emb_e_real(e2).reshape(batch_size, -1)
            e2_embedded_img = self.emb_e_img(e2).reshape(batch_size, -1)
            realrealreal = e1_embedded_real * rel_embedded_real * e2_embedded_real
            realimgimg = e1_embedded_real * rel_embedded_img * e2_embedded_img
            imgrealimg = e1_embedded_img * rel_embedded_real * e2_embedded_img
            imgimgreal = e1_embedded_img * rel_embedded_img * e2_embedded_real
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            pred = F.sigmoid(F.sum(pred, 1))
            return pred
        else:
            realrealreal = F.matmul(e1_embedded_real * rel_embedded_real, self.emb_e_real.W, transb=True)
            realimgimg = F.matmul(e1_embedded_real * rel_embedded_img, self.emb_e_img.W, transb=True)
            imgrealimg = F.matmul(e1_embedded_img * rel_embedded_real, self.emb_e_img.W, transb=True)
            imgimgreal = F.matmul(e1_embedded_img * rel_embedded_img, self.emb_e_real.W, transb=True)
            pred = realrealreal + realimgimg + imgrealimg - imgimgreal
            pred = F.sigmoid(pred)
            return pred


class ConvE(chainer.Chain, BaseModel):
    """
    Convolutional 2D Knowledge Graph Embeddings, Tim Dettmers et al., 2017
    """
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = 200

        with self.init_scope():
            self.emb_e = L.EmbedID(
                    num_entities, self.embedding_dim, socketinitialW=I.GlorotNormal())
            self.emb_rel = L.EmbedID(
                    num_relations, self.embedding_dim, initialW=I.GlorotNormal())
            self.conv1 = L.Convolution2D(1, 32, 3, stride=1, pad=0)
            self.bias = L.EmbedID(num_entities, 1)
            self.fc = L.Linear(10368, self.embedding_dim)
            self.bn0 = L.BatchNormalization(1)
            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(self.embedding_dim)

    def forward(self, e1, rel, e2=None):
        """
        Input:
            e1, rel, e2: ids for each entity and relation (shape : (batchsize,) )
        Output:
            score (shape : (batchsize,) )
        """
        batch_size, = e1.shape
        e1_embedded = self.emb_e(e1).reshape(batch_size, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).reshape(batch_size, 1, 10, 20)

        stacked_inputs = F.concat([e1_embedded, rel_embedded], axis=2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = F.dropout(stacked_inputs, 0.2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        x = F.dropout(x, 0.3)
        x = self.bn2(x)
        x = F.relu(x)
        if e2 is not None:
            e2_embedded = self.emb_e(e2)
            bias = self.bias(e2).reshape((-1,))
            x *= e2_embedded
            x = F.sum(x, axis=1) + bias
            pred = F.sigmoid(x)
            return pred
        else:
            x = F.matmul(x, self.emb_e.W, transb=True)
            x, bias = F.broadcast(x, self.bias.W.T)
            x += bias
            pred = F.sigmoid(x)
            return pred

    def __call__(self, e1, rel, e2, Y, flt):
        """
        Input:
            e1, rel, e2: ids for each entity and relation (shape : (batchsize,) )
            Y: whether true (1) or negative (0) sample (shape : (batchsize,) )
            flt: {0, 1} array used for filter evaluation (shape : (batch_size, num_entities) )
        Output:
            loss ( float )
        """
        if chainer.config.train:
            probs = self.forward(e1, rel)
            probs = probs.reshape((-1,))
            loss = self.binary_cross_entropy(probs, Y)
            reporter.report({'loss': loss}, self)
            return loss
        else:
            assert flt is not None
            batch_size, = e1.shape
            probs_all = self.forward(e1, rel, None)
            metrics = self.evaluate(probs_all, e2, flt)
            probs = probs_all[self.xp.arange(batch_size), e2]
            loss = self.binary_cross_entropy(probs, Y)
            metrics['loss'] = loss
            reporter.report(metrics, self)
            return loss


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
        return self.word2id[word]

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    @classmethod
    def load(cls, vocab_path):
        v = Vocab()
        with open(vocab_path) as f:
            for word in f:
                v.add(word.strip())
        return v


class TripletDataset(chainer.dataset.DatasetMixin):
    def __init__(self, ent_vocab, rel_vocab, path, negative, flt_graph=None):
        self.path = path
        logger.info("creating TripletDataset for: {}".format(self.path))
        self.negative = negative
        self.entities = ent_vocab
        self.relations = rel_vocab
        self.data = []
        if flt_graph is not None:
            logger.info("filtered on")
            self.filtered = True
            self.graph = flt_graph
        else:
            logger.info("filtered off")
            self.filtered = False
            self.graph = defaultdict(list)
        self.load_from_path()

    def __len__(self):
        return len(self.data)

    def load_from_path(self):
        logger.info("start loading dataset")
        for line in open(self.path):
            e1, rel, e2 = line.strip().split("\t")
            id_e1 = self.entities[e1]
            id_e2 = self.entities[e2]
            id_rel = self.relations[rel]
            self.data.append((id_e1, id_rel, id_e2))
            self.graph[id_e1, id_rel].append(id_e2)
        logger.info("done")
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)
        logger.info("num samples: {}".format(len(self)))
        logger.info("num entities: {}".format(self.num_entities))
        logger.info("num relations: {}".format(self.num_relations))

    def get_example(self, i):
        triplet = self.data[i]
        triplets = np.asarray(triplet * (1 + self.negative), 'i').reshape((-1, 3))
        neg_ents = np.random.randint(0, self.num_entities, size=self.negative)
        head_or_tail = 2 * np.random.randint(0, 2, size=self.negative)
        triplets[np.arange(1, self.negative+1), head_or_tail] = neg_ents
        e1, rel, e2 = zip(*triplets)
        Y = np.zeros(1 + self.negative, 'i')
        Y[0] = 1
        if self.filtered:
            e1_id, rel_id, e2_id = triplet
            flt = np.ones(self.num_entities, 'f')
            flt[self.graph[e1_id, rel_id]] = 0.
            flt[e2_id] = 1.
        else:
            flt = None
        return e1, rel, e2, Y, flt


class FastEvalTripletDataset(chainer.dataset.DatasetMixin):
    """
    Dataset to perform the technique in:
      4.1 Fast Evaluation for Link Prediction Tasks
      (Convolutional 2D Knowledge Graph Embeddings, Tim Dettmers et al.)
    """
    def __init__(self, ent_vocab, rel_vocab, path):
        self.path = path
        logger.info("creating FastEvalTripletDataset for: {}".format(self.path))
        self.entities = ent_vocab
        self.relations = rel_vocab
        self.data = []
        self.graph = defaultdict(list)
        self.deepgraph = defaultdict(list)
        self.load_from_path()

    def __len__(self):
        return len(self.data)

    def load_from_path(self):
        logger.info("start loading dataset")
        for line in open(self.path):
            e1, rel, e2 = line.strip().split("\t")
            id_e1 = self.entities[e1]
            id_e2 = self.entities[e2]
            id_rel = self.relations[rel]
            self.graph[id_e1, id_rel].append(id_e2)
        self.data.extend(list(self.graph.items()))
        logger.info("done")
        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)
        logger.info("num samples: {}".format(len(self)))
        logger.info("num entities: {}".format(self.num_entities))
        logger.info("num relations: {}".format(self.num_relations))

        trans = [self.relations[s] for s in ["hypernyms", "hyponyms"]]
        for id_e1, id_rel in tqdm(self.graph, desc="extending graph"):
            if id_rel not in trans:
                self.deepgraph[id_e1, id_rel] = self.graph[id_e1, id_rel]
                continue
            res = []

            def traverse(e1, rel, depth):
                res.append(e1)
                if depth <= 0: return
                if (e1, rel) in self.graph:
                    for e2 in self.graph[e1, rel]:
                        traverse(e2, rel, depth - 1)

            for e2 in self.graph[id_e1, id_rel]:
                traverse(e2, id_rel, 50)
            self.deepgraph[id_e1, id_rel] = list(set(res))
        self.graph = self.deepgraph
        self.data = list(self.deepgraph.items())

    def get_example(self, i):
        (e1, rel), e2 = self.data[i]
        e1 = np.array([e1], 'i')
        rel = np.array([rel], 'i')
        e2 = np.array(e2, 'i')
        Y = np.zeros(self.num_entities, 'i')
        Y[e2] = 1
        flt = None
        return e1, rel, e2, Y, flt


def convert(batch, device):
    e1, rel, e2, Y, flt = zip(*batch)
    e1 = np.concatenate(e1)
    rel = np.concatenate(rel)
    e2 = np.concatenate(e2)
    Y = np.concatenate(Y)
    flt = np.vstack(flt) if flt[0] is not None else None
    if device >= 0:
        e1 = cuda.to_gpu(e1)
        rel = cuda.to_gpu(rel)
        e2 = cuda.to_gpu(e2)
        Y = cuda.to_gpu(Y)
        if flt is not None:
            flt = cuda.to_gpu(flt)
    return e1, rel, e2, Y, flt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Path to training triplet list file')
    parser.add_argument('val', help='Path to validation triplet list file')
    parser.add_argument('ent_vocab', help='Path to entity vocab')
    parser.add_argument('rel_vocab', help='Path to relation vocab')
    parser.add_argument('--model', default='conve', choices=['complex', 'conve'])
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=1000, help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to learn')
    parser.add_argument('--negative-size', default=10, type=int, help='number of negative samples')
    parser.add_argument('--out', default='result', help='Directory to output the result')
    parser.add_argument('--val-iter', type=int, default=1000, help='validation iteration')
    parser.add_argument('--init-model', default=None, help='initialize model with saved one')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)
    log_path = os.path.join(args.out, 'loginfo')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info('train: {}'.format(args.train))
    logger.info('val: {}'.format(args.val))
    logger.info('gpu: {}'.format(args.gpu))
    logger.info('model: {}'.format(args.model))
    logger.info('batchsize: {}'.format(args.batchsize))
    logger.info('epoch: {}'.format(args.epoch))
    logger.info('negative-size: {}'.format(args.negative_size))
    logger.info('out: {}'.format(args.out))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    ent_vocab = Vocab.load(args.ent_vocab)
    rel_vocab = Vocab.load(args.rel_vocab)

    if args.model == 'conve':
        train = FastEvalTripletDataset(ent_vocab, rel_vocab, args.train)
    else:
        train = TripletDataset(ent_vocab, rel_vocab, args.train, args.negative_size)
    val = TripletDataset(ent_vocab, rel_vocab, args.val, 0, train.graph)

    if args.model == 'conve':
        model = ConvE(train.num_entities, train.num_relations)
    elif args.model == 'complex':
        model = ComplEx(train.num_entities, train.num_relations)
    else:
        raise "no such model available: {}".format(args.model)

    if args.init_model:
        logger.info("initialize model with: {}".format(args.init_model))
        serializers.load_npz(args.init_model, model)
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False)

    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    val_interval = args.val_iter, 'iteration'
    log_interval = 100, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model,
        converter=convert, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss',
        'validation/main/loss', 'validation/main/mrr', 'validation/main/mrr(flt)',
        'validation/main/hits1(flt)', 'validation/main/hits3(flt)',
        'validation/main/hits10(flt)']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()

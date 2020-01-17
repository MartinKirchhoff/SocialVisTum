import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gensim
import codecs
import argparse
import multiprocessing
from collections import Counter
import logging
from scipy import sparse
import numpy as np
import pickle
from utils import createPath
from utils import listify
from mittens import Mittens

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")


class Sentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def w2v(source, emb_dim):
    sentences = Sentences(source)
    model = gensim.models.Word2Vec(sentences, iter=15, size=emb_dim, window=10, min_count=5,
                                   workers=multiprocessing.cpu_count())
    return model


class Glove(object):
    def build_vocab(self, corpus):
        logger.info("Building vocab from corpus")
        vocab = Counter()
        for line in corpus:
            tokens = line.strip().split()
            vocab.update(tokens)
        vocab = vocab.most_common(18000)
        logger.info("Done building vocab {} from corpus.".format(18000))

        return {word: (i, freq) for i, (word, freq) in enumerate(vocab)}

    # @listify
    def build_cooccur(self, vocab, corpus, window_size=10, min_count=None):

        vocab_size = len(vocab)
        id2word = dict((i, word) for word, (i, _) in vocab.items())

        cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                          dtype=np.float64)

        for i, line in enumerate(corpus):
            if i % 1000 == 0:
                logger.info("Building cooccurrence matrix: on line %i", i)

            tokens = line.strip().split()
            token_ids = [vocab[word][0] for word in tokens if word in vocab]

            for center_i, center_id in enumerate(token_ids):
                # Collect all word IDs in left window of center word
                context_ids = token_ids[max(0, center_i - window_size): center_i]
                contexts_len = len(context_ids)

                for left_i, left_id in enumerate(context_ids):
                    # Distance from center word
                    distance = contexts_len - left_i

                    # Weight by inverse of distance between words
                    increment = 1.0 / float(distance)

                    # Build co-occurrence matrix symmetrically (pretend we
                    # are calculating right contexts as well)
                    cooccurrences[center_id, left_id] += increment
                    cooccurrences[left_id, center_id] += increment

        for i, (row, data) in enumerate(zip(cooccurrences.rows, cooccurrences.data)):
            if min_count is not None and vocab[id2word[i]][1] < min_count:
                continue

            for data_idx, j in enumerate(row):
                if min_count is not None and vocab[id2word[j]][1] < min_count:
                    continue

                yield i, j, data[data_idx]

    def get_original_embedding(self, filename):
        embed_dict = {}
        fin = codecs.open(filename, 'r', 'utf-8')
        for line in fin.readlines():
            splitted_line = line.strip().split()
            word = splitted_line[0]
            embedding = np.asarray(splitted_line[1:], dtype=np.float64)
            embed_dict[word] = embedding
        return embed_dict

    def convert_cooccurence_matrix(self, cooccurence, vocab_size):
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for c in cooccurence:
            c = np.array(c, dtype=np.float64).tolist()
            co_matrix[int(c[0]), int(c[1])] = c[2]
        return co_matrix

    def load_vocab_in_order(self, vocab_file):
        file1 = open(vocab_file, 'rb')
        vocab_word = pickle.load(file1)
        file1.close()
        logger.info('Vocab file loaded')
        vocab_word_x = sorted(vocab_word.items(), key=lambda x: x[1][0])
        vocab = []
        for (w, (i, _)) in vocab_word_x:
            vocab.append(w)
        logger.info('Check vocab order: {}, {}'.format(vocab[0], vocab[1]))
        return vocab

    def load_cooccurence_matrix(self, cooccurence_file):
        file1 = open(cooccurence_file, 'rb')
        cooccurence = pickle.load(file1)
        file1.close()
        logger.info('Cooccurence to matrix of shape: {}'.format(cooccurence.shape))
        return cooccurence

    def load_glove_embedding(self, filename):
        file1 = open(filename, 'rb')
        glove_embedding = pickle.load(file1)
        file1.close()
        logger.info('glove embedding loaded')
        return glove_embedding

    def convert_glove_2_w2v(self, vocab_file, embedding_file, output_file):
        glove_embedding = self.load_glove_embedding(embedding_file)
        vocab = self.load_vocab_in_order(vocab_file)
        fout = codecs.open(output_file, 'w', 'utf-8')
        (vocab_no, dim) = glove_embedding.shape
        fout.write('{} {}'.format(vocab_no, dim) + '\n')
        for i, embedding in enumerate(glove_embedding):
            c = np.array(embedding, dtype=np.float64).tolist()
            c = " ".join(map(str, c))
            fout.write((vocab[i].encode('utf-8') + b" " + c.encode('utf-8') + b'\n').decode('utf-8'))
        fout.close()

    def convert_glove_2_w2v_type2(self, embedding_file, output_file):
        fin = codecs.open(embedding_file, "r", "utf-8")
        fout = codecs.open(output_file, "w", "utf-8")
        lines = [line.strip() for line in fin]
        fin.close()
        vocab_no = len(lines)
        dim = len(lines[0].split()[1:])
        fout.write("{} {}".format(vocab_no, dim) + "\n")
        for line in lines:
            fout.write(line + "\n")
        fout.close()


def glove_embedding(filename, vocab_file, cooccurence_file, domain):
    gv = Glove()
    out_dir = './preprocessed_data/' + domain
    if vocab_file and cooccurence_file:
        vocab = gv.load_vocab_in_order(vocab_file)
        cooccurence = gv.load_cooccurence_matrix(cooccurence_file)
        logger.info('get pre-trained glove embedding')
        original_embedding = gv.get_original_embedding('./pretrained_embeddings/glove.6B/glove.6B.300d.txt')
        mittens_model = Mittens(n=300, max_iter=1000)
        logger.info('Start fine tuning...')
        new_embeddings = mittens_model.fit(cooccurence, vocab=vocab,
                                           initial_embedding_dict=original_embedding)
        fin = open(out_dir + '/fine_tuned_glove_300', 'wb')
        pickle.dump(new_embeddings, fin)
        fin.close()
        logger.info('Fine tuning complete')
    else:
        logger.info('Load english data')
        fin = codecs.open(filename, 'r', 'utf-8')
        corpus = []
        for line in fin:
            corpus.append(line)
        vocab = gv.build_vocab(corpus)
        vocab_file = out_dir + '/vocab.pkl'
        createPath(vocab_file)
        outfile = open(vocab_file, 'wb')
        pickle.dump(vocab, outfile)
        outfile.close()
        logger.info("Fetching cooccurrence list..")
        cooccurrences = gv.build_cooccur(vocab, corpus)
        cooccurrences = gv.convert_cooccurence_matrix(cooccurrences, len(vocab))
        cooccurrence_file = out_dir + '/cooccurrence.pkl'
        outfile = open(cooccurrence_file, 'wb')
        pickle.dump(cooccurrences, outfile)
        outfile.close()
        logger.info("Cooccurrence list fetch complete (%i pairs).\n", cooccurrences.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-emb_dim", "--emb_dim", dest="emb_dim", type=int, default=300,
                        help="embedding dimension value")
    parser.add_argument("-sf", "--source-file", dest="source_file", type=str, metavar='<str>',
                        help="Name the source file")
    parser.add_argument("-gv_vocab", "--glove_vocab", dest="glove_vocab", type=str, metavar='<str>',
                        help="Name the vocab file")
    parser.add_argument("-gv_cooccur", "--gv_cooccur", dest="glove_cooccurence", type=str, metavar='<str>',
                        help="Name the cooccurence file")
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='organic',
                        help="domain of the corpus")
    args = parser.parse_args()

    logger.info('Generating glove embedding...')
    glove_embedding(args.source_file, args.glove_vocab, args.glove_cooccurence, args.domain)


if __name__ == "__main__":
    main()

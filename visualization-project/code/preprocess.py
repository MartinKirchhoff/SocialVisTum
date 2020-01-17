from __future__ import print_function
import os
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import argparse
import utils as U
from word_embedding import Glove


def parseSentence(line):
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


def preprocess_train(domain):
    f = codecs.open('./datasets/' + domain + '/train.txt', 'r', 'utf-8')

    U.mkdir_p('./preprocessed_data/' + domain)

    out = codecs.open('./preprocessed_data/' + domain + '/train.txt', 'w', 'utf-8')
    num_train_sen = 0
    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 1:
            out.write(' '.join(tokens) + '\n')
            num_train_sen += 1

    print("Number of training sentences:", num_train_sen)


def preprocess_test(domain):
    f1 = codecs.open('./datasets/' + domain + '/test.txt', 'r', 'utf-8')
    f2 = codecs.open('./datasets/' + domain + '/test_labels.txt', 'r', 'utf-8')

    out1 = codecs.open('./preprocessed_data/' + domain + '/test.txt', 'w', 'utf-8')
    out2 = codecs.open('./preprocessed_data/' + domain + '/test_labels.txt', 'w', 'utf-8')

    num_test_sen = 0
    for text, label in zip(f1, f2):
        label = label.strip()

        tokens = parseSentence(text)
        if len(tokens) > 1:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label + '\n')
            num_test_sen += 1
    print("Number of test sentences:", num_test_sen)


def preprocess(domain):
    print('\t' + domain + ' train set ...')
    preprocess_train(domain)
    print('\t' + domain + ' test set ...')
    preprocess_test(domain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--domain-name", dest="domain_name", type=str, metavar='<str>', required=True,
                        help="The domain name")
    args = parser.parse_args()

    print('Preprocessing raw review sentences ...')
    nltk.download('wordnet')
    nltk.download('stopwords')
    preprocess(args.domain_name)

    os.system(
        "python ./code/word_embedding.py -sf ./preprocessed_data/" + args.domain_name + "/train.txt" + " --domain " + args.domain_name)
    os.system(
        "python ./code/word_embedding.py -sf ./preprocessed_data/" + args.domain_name + "/train.txt" + " -gv_vocab ./preprocessed_data/" + args.domain_name + "/vocab.pkl" + " -gv_cooccur " + "./preprocessed_data/" + args.domain_name + "/cooccurrence.pkl" + " --domain " + args.domain_name)
    gv = Glove()
    gv.convert_glove_2_w2v('./preprocessed_data/' + args.domain_name + '/vocab.pkl',
                           './preprocessed_data/' + args.domain_name + '/fine_tuned_glove_300',
                           './preprocessed_data/' + args.domain_name + '/glove_w2v_300_filtered.txt')


if __name__ == "__main__":
    main()

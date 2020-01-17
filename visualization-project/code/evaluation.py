from __future__ import print_function
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import pandas as pd
from sklearn.metrics import classification_report
import codecs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import scipy.cluster.hierarchy as shc
from itertools import permutations
from sklearn.metrics import f1_score
import random
from keras.preprocessing import sequence
import reader as dataset
from model import create_model
import keras.backend as K
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class Evaluation(object):
    """Class used to evaluate the performance based on the F1 score.

       Args:
            args: Argparse instance that contains the relevant parameters
            test_labels: List of test labels that are used for the F1 score calculation
        """

    def __init__(self, args, test_labels):
        self.args = args
        self.test_labels = test_labels
        self.out_dir = './code/output_dir/' + args.domain + "/" + args.conf
        self.topic_probs = None

    def build_model(self):
        """Creates the model object, which is used to calculate F1 scores

        Returns:

        """

        # Get test data
        vocab, train_x, test_x, overall_maxlen = dataset.get_data(self.args.domain, vocab_size=self.args.vocab_size,
                                                                  maxlen=self.args.maxlen)

        input_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

        # Create Model
        model = create_model(self.args, overall_maxlen, vocab)
        logger.info('Model initialized')

        # Load weights
        model.load_weights(self.out_dir + '/model_param')

        test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                             [model.get_layer('att_weights').output, model.get_layer('p_t').output])

        # Calculate topic probabilities
        att_weights, topic_probs = test_fn([input_x, 0])
        logger.info('Topic Probabilities calculated')

        # Save probabilities to file
        np.savetxt(self.out_dir + '/topic_probs.txt', topic_probs, fmt='%f')
        self.topic_probs = topic_probs

    def general_baseline(self):
        """Calculates the F1 score when mapping every topic to "general".

        Returns:
            cluster_map: Dictionary that maps every topic to "general"

        """
        cluster_map = {}
        for cluster_number in range(self.args.num_topics):
            cluster_map[cluster_number] = "general"
        return cluster_map

    def random_mapping(self, n_iterations=1000):
        """Calculates a random mapping between the topics and the coarse attributes. The mean F1 score is returned.

        Args:
            n_iterations: Number of random iterations.

        Returns:
            Float that sepcifies the average F1 score

        """
        keys = [i for i in range(0, self.args.num_topics)]
        cluster_map = dict.fromkeys(keys)
        f1_scores = []
        for iteration in range(n_iterations):
            for key in cluster_map:
                cluster_map[key] = random.choice(self.test_labels)
            f1_scores.append(self.get_f1_scores(cluster_map))
        return round(sum(f1_scores) / len(f1_scores), 3)

    # Calculate the best possible mapping between the clusters and the actual test labels
    def best_manual_mapping(self):
        """Calculates the best possible mapping (based on F1 score) between topics and coarse attributes.

        Returns:
            cluster_map: Dictionary that maps the topics to suitable coarse attributes

        """

        # Initialize random mapping
        keys = [i for i in range(0, self.args.num_topics)]
        cluster_map = dict.fromkeys(keys)
        for key in cluster_map:
            cluster_map[key] = random.choice(self.test_labels)

        # Iterate through all possible test labels and choose the label with the best F1 score for each key
        for key in range(self.args.num_topics):
            f1_scores = []
            for poss_val in self.test_labels:
                cluster_map[key] = poss_val
                f1_scores.append(self.get_f1_scores(cluster_map))
            cluster_map[key] = self.test_labels[f1_scores.index(max(f1_scores))]
        return cluster_map

    def prediction(self, topic_mapping):
        """Create the classification report for a given mapping.

        Args:
            topic_mapping: Dictionary that contains the mapping to coarse attributes.

        Returns:
            Classification report that shows the F1 Scores for all classes

        """
        specific_df = pd.DataFrame(self.topic_probs, columns=topic_mapping.values())
        predict_labels = specific_df.idxmax(axis=1)
        test_file = open('./preprocessed_data/%s/test_labels.txt' % (self.args.domain))

        true_label_li = []
        for line in test_file:
            true_label_li.append(line.strip())
        return classification_report(true_label_li, predict_labels, self.test_labels, digits=3)

    def get_f1_scores(self, topic_mapping, ignore_general=False):
        """Calculates the F1 scores for a given mapping.

        Args:
            topic_mapping: Dictionary that contains the mapping
            ignore_general: Boolean that specifies whether the "general" coarse attribute should be ignored or not

        Returns:
            F1 score for the given mapping

        """
        specific_df = pd.DataFrame(self.topic_probs, columns=topic_mapping.values())
        predict_labels = specific_df.idxmax(axis=1)
        test_file = open('./preprocessed_data/%s/test_labels.txt' % (self.args.domain))
        true_label_li = []
        for line in test_file:
            true_label_li.append(line.strip())
        if ignore_general:
            return f1_score(true_label_li, predict_labels, self.test_labels, average=None)
        else:
            return f1_score(true_label_li, predict_labels, self.test_labels, average=self.args.f1_metric)

    def k_means_clustering(self, hierarchical=False, plot_dendogram=False, possible_maps=None):
        """Applies k-means or hierarchical clustering.

        Args:
            hierarchical: Boolean that specifies whether the clustering should be hierarchical
            plot_dendogram: Boolean that specifies whether a dendogram for the hierarchical clustering should be shown
            possible_maps: List of all mappings

        Returns:
            Dictionary that contains the mapping based on k-means or hierarchical clusteirng

        """
        df = pd.DataFrame(self.topic_probs)
        corr_df = df.corr(method="pearson")
        corr_array = corr_df.values

        if hierarchical:
            cluster_map = {0: 'general', 1: 'general', 2: 'safety and healthiness', 3: 'environment',
                           4: 'experienced quality',
                           5: 'general'}
            num_clusters = len(cluster_map)
            cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='single').fit(
                corr_array)

            if plot_dendogram:
                self.show_dendogram(corr_array)
        else:
            cluster_map = {0: 'safety and healthiness', 1: 'general', 2: 'trustworthy sources', 3: 'general',
                           4: 'experienced quality',
                           5: 'environment'}
            num_clusters = len(cluster_map)
            cluster = KMeans(n_clusters=num_clusters, random_state=42).fit(corr_array)

        best_mapping = {}

        if possible_maps is None:
            for cluster_number in range(num_clusters):
                indices = np.where(cluster.labels_ == cluster_number)[0]
                for idx in indices:
                    best_mapping[idx] = cluster_map[cluster_number]
        else:
            best_mapping = self.calculate_best_mapping(cluster, possible_maps, True,
                                                       "Hierarchical=" + str(hierarchical))
        return best_mapping

    def show_dendogram(self, corr_array):
        """Shos the dendogram for a given correlation array.

        Args:
            corr_array: Numpy array that contains the correlations between the topics

        Returns:

        """
        cluster_labels = open(self.out_dir + '/node_names.log', "r").read().splitlines()
        plt.figure(figsize=(18, 10))
        plt.title("Clustering Dendogram")
        generic_labels = ["Topic_" + str(index) + "\n" + cluster_labels[index] for index in
                          range(len(cluster_labels))]
        dend = shc.dendrogram(shc.linkage(corr_array, method='single'), labels=generic_labels, leaf_font_size=6)
        plt.show()

    def calculate_best_mapping(self, cluster, possible_maps, plot_f1_scores=True, plot_name="",
                               ignore_gen_and_sources=False):
        """Calculates the best mapping for a list of possible mappings

        Args:
            cluster: Sklearn cluster object
            possible_maps: List of all mappings
            plot_f1_scores: Boolean that specifies whether the F1 score should be plotted
            plot_name: String that specifies the name of the plot title
            ignore_gen_and_sources: Boolean that specifies whether the coarse attributes "general" and "trustworthy sources" should be ignored or not

        Returns:
            best_mapping: Dictionary that contains the best mapping

        """
        best_f1_score = 0
        f1_scores = []
        for curr_map in possible_maps:
            inverted_map = {}
            for cluster_number in range(len(cluster.labels_)):
                indices = np.where(cluster.labels_ == cluster_number)[0]
                for idx in indices:
                    inverted_map[idx] = curr_map[cluster_number]
            if ignore_gen_and_sources:
                shortened_f1_score = self.get_f1_scores(inverted_map, ignore_general=True)[2:]
                curr_f1_score = round(sum(shortened_f1_score) / len(shortened_f1_score), 3)
            else:
                curr_f1_score = self.get_f1_scores(inverted_map)
            f1_scores.append(curr_f1_score)
            if curr_f1_score > best_f1_score:
                best_mapping = inverted_map
                best_f1_score = curr_f1_score
        if plot_f1_scores:
            self.plot_f1_score(f1_scores, plot_name)

        return best_mapping

    def plot_f1_score(self, values, title=""):
        """Plots the F1 score distribution given a number of values.

        Args:
            values: List that contains the F1 score values
            title: String that specifies the title of the plot

        Returns:

        """
        plt.figure()
        sns.set(rc={'figure.figsize': (12, 6)})
        ax = sns.distplot(values, hist=False)
        plt.xlabel("F1 Score")
        plt.ylabel("Occurences")
        plt.title("F1 Score Occurences " + title)
        plt.savefig(self.out_dir + "/f1_score_" + title + ".png")

    def get_all_possible_mappings(self):
        """Retrieves all possible mappings of the coarse attributes.

        Returns:
            maps: List of all mappings that are possible given the coarse attributes

        """
        combs = list(permutations(self.test_labels, len(self.test_labels)))
        maps = []
        for comb_idx in range(len(combs)):
            map = {}
            comb = combs[comb_idx]
            for idx in range(len(comb)):
                map[idx] = comb[idx]
            maps.append(map)
        return maps

    def write_mapping_results(self, clustering_names, clustering_methods):
        """Writes the mapping results to a new file.

        Args:
            clustering_names: List that contains the name for each mapping (can be any String)
            clustering_methods: List that contains the mappings (dictionaries)

        Returns:

        """
        result_file = codecs.open(self.out_dir + '/results', 'w', 'utf-8')
        result_file.write("Optimizing " + self.args.f1_metric + " F1 score" + "\n" + "\n")

        for ind in range(len(clustering_methods)):
            pred = self.prediction(clustering_methods[ind])
            result_file.write(clustering_names[ind] + ":" + "\n")
            result_file.write(str(clustering_methods[ind]) + "\n" + "\n")
            result_file.write(pred + "\n")

            shortened_f1_score = self.get_f1_scores(clustering_methods[ind], ignore_general=True)[2:]
            curr_f1_score = round(sum(shortened_f1_score) / len(shortened_f1_score), 3)
            result_file.write(
                "F1 score (ignoring general and trustworthy sources): " + str(curr_f1_score) + "\n" + "\n")

        result_file.write("Random mapping F1 score: " + str(self.random_mapping()))
        logger.info('Results written to result file')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300,
                        help="Embeddings dimension (default=300)")
    parser.add_argument("-wv_type", "--wordvec-type", dest="wv_type", type=str, metavar='<str>',
                        default="glove_finetuned",
                        help="The type of word vectors to use (glove, both, w2v, word2vec_finetune, glove_finetuned)")
    parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                        help="The weight of orthogonol regularizaiton (default=0.1)")
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000,
                        help="Vocab size. '0' means no limit")
    parser.add_argument("-as", "--num-topics", dest="num_topics", type=int, metavar='<int>', default=50,
                        help="The number of topics specified by users")
    parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',
                        help="The path to the word embeddings file")
    parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                        help="Number of negative instances (default=20)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                        help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234,
                        help="Random seed (default=1234)")
    parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                        help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', required=True,
                        help="domain of the corpus")
    parser.add_argument("--conf", dest="conf", type=str, metavar='<str>', required=True,
                        help="test configuration for the given domain")
    parser.add_argument("--f1-metric", dest="f1_metric", type=str, metavar='<str>', default="micro",
                        help="F1 Metric used (micro, macro)")
    parser.add_argument("--fix_clusters", dest="fix_clusters", type=str, metavar='<str>', default="no",
                        help="To fix initial clusters (yes or no)")

    args = parser.parse_args()
    # U.print_args(args)

    assert args.f1_metric in {'micro', 'macro'}

    test_labels = ["general", "trustworthy sources", "safety and healthiness", "environment", "experienced quality",
                   "price"]
    evaluator = Evaluation(args, test_labels)
    evaluator.build_model()
    general_baseline = evaluator.general_baseline()
    best_mapping = evaluator.best_manual_mapping()

    evaluator.write_mapping_results(["Best Mapping", "General Baseline"], [best_mapping, general_baseline])


if __name__ == "__main__":
    main()

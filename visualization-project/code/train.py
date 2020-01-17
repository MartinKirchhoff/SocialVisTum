from __future__ import print_function
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import numpy as np
from time import time
import utils as U
import codecs
from optimizers import get_optimizer
from model import create_model
import keras.backend as K
from keras.preprocessing import sequence
import reader as dataset
from tqdm import tqdm
import pandas as pd
import json
from nltk.corpus import wordnet as wn
from collections import Counter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Node(object):
    """Class represents the node objects, which are displayed in the JSON file.

       Args:
            id: String that specifies the topic label
            group: Integer that specifies the color of the node
            occurrences: String that specifies the number of topic occurrences
            words: Lists of representative words
            sentences: List of representative sentences
        """

    def __init__(self, id, group, occurrences, words, sentences):
        self.id = id
        self.group = group
        self.occurrences = occurrences
        self.words = words
        self.sentences = sentences

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=2)


class Link(object):
    """Class represents the link objects, which are displayed in the JSON file.

       Args:
            source: String that specifies the topic label of the first node, the link is connected to
            target: String that specifies the topic label of the second node, the link is connected to
            value: Float that specifies the similarity (correlation) between source and target
        """

    def __init__(self, source, target, value):
        self.source = source
        self.target = target
        self.value = value

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=2)


class TopWord(object):
    """Class represents top words that are displayed in the JSON file

       Args:
            rank: Integer that specifies the rank in the word list (e.g., 1 --> Most representative word)
            word: Unicode that specifies the word
            similarity: String that specifies the similarity (correlation) to the topic embedding
        """

    def __init__(self, rank, word, similarity):
        self.rank = rank
        self.word = word
        self.similarity = similarity


class TopSentence(object):
    """Class represents top sentences that are displayed in the JSON file

       Args:
            rank: Integer that specifies the rank in the sentence list (e.g., 1 --> Most representative sentence)
            sentence: Unicode that specifies the sentence
        """

    def __init__(self, rank, sentence):
        self.rank = rank
        self.sentence = sentence


class Train(object):
    """Class used to train the model and generate relevant topic information

       Args:
            args: Argparse instance that contains the relevant parameters
            logger: Logger instance
            out_dir: String that contains the path to the output directory
        """

    def __init__(self, args, logger, out_dir):
        self.args = args
        self.logger = logger
        self.out_dir = out_dir
        self.vocab, train_x, test_x, self.overall_maxlen = dataset.get_data(self.args.domain,
                                                                            vocab_size=self.args.vocab_size,
                                                                            maxlen=self.args.maxlen)
        self.train_x = sequence.pad_sequences(train_x, maxlen=self.overall_maxlen)
        self.test_x = sequence.pad_sequences(test_x, maxlen=self.overall_maxlen)
        self.vis_path = self.out_dir + "/visualization"
        U.mkdir_p(self.vis_path)

    def sentence_batch_generator(self, data, batch_size):
        """ Generates batches based on the data.

        Args:
            data: Numpy array of the data
            batch_size: Integer that specifies the batch size (e.g. 64)

        Returns:

        """
        n_batch = len(data) / batch_size
        batch_count = 0
        np.random.shuffle(data)

        while True:
            if batch_count == n_batch:
                np.random.shuffle(data)
                batch_count = 0

            batch = data[batch_count * batch_size: (batch_count + 1) * batch_size]
            batch_count += 1
            yield batch

    def negative_batch_generator(self, data, batch_size, neg_size):
        """Generates negative batches based on the data.

        Args:
            data: Numpy array of the data
            batch_size: Integer that specifies the batch size (e.g. 64)
            neg_size: Integer that specifies the number of negative instances

        Returns:

        """
        data_len = data.shape[0]
        dim = data.shape[1]

        while True:
            indices = np.random.choice(data_len, batch_size * neg_size)
            samples = data[indices].reshape(batch_size, neg_size, dim)
            yield samples

    def write_topics(self, word_emb, topic_emb, epoch, vocab_inv):
        """Writes relevant topic information with similar words to .log file for each epoch.

        Args:
            word_emb: Numpy array that contains the word embeddings
            topic_emb: Numpy array that contains the topic embeddings
            epoch: Integer that specifies the current epoch
            vocab_inv: Dictionary that maps the index of every word in the vocab file to the corresponding word (In descending order based on occurrences)

        Returns:

        """

        # In final epoch, write in main directory
        if epoch == self.args.epochs:
            topic_file = codecs.open(self.out_dir + '/topics.log', 'w', 'utf-8')
        # In other epochs, write in subdirectory
        else:
            topic_file = codecs.open(self.out_dir + '/topics/topic_epoch_' + str(epoch) + '.log', 'w', 'utf-8')

        # Get the most similar words for every topic
        for topic in range(self.args.num_topics):
            desc = topic_emb[topic]
            sims = word_emb.dot(desc.T)
            ordered_words = np.argsort(sims)[::-1]
            found_words = 0
            desc_list = []

            # Save most similar words until enough words are found
            for word in ordered_words:
                if found_words == self.args.labeling_num_words:
                    break

                elif vocab_inv[word] != "<unk>":
                    # Save word and associated similarity
                    desc_list.append(vocab_inv[word] + "|" + str(sims[word]))
                    found_words += 1

            # Write most similar words to file
            topic_file.write('Topic %d:\n' % topic)
            topic_file.write(' '.join(desc_list) + '\n\n')

    # Returns a dataframe containing the most similar words for every topic and a list containing the topic names
    def get_similar_words(self, model, vocab_inv):
        """
        Args:
            model: Keras model object
            vocab_inv: Dictionary that maps the index of every word in the vocab file to the corresponding word (In descending order based on occurrences)

        Returns:
            topic_labels: Lists that contains the topic names (Based on selecting the most similar word)
            word_df: DataFrame that contains the most similar words of every topic

        """

        # Get all word and topic embeddings
        word_emb = K.get_value(model.get_layer('word_emb').embeddings)
        word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
        topic_emb = K.get_value(model.get_layer('topic_emb').W)
        topic_emb = topic_emb / np.linalg.norm(topic_emb, axis=-1, keepdims=True)

        word_df = pd.DataFrame(columns=['topic', 'rank', 'word', "similarity"])
        topic_labels = []

        # Iterate through every topic and calculate the most similar words
        for topic in range(self.args.num_topics):
            desc = topic_emb[topic]
            sims = word_emb.dot(desc.T)
            ordered_words = np.argsort(sims)[::-1]
            found_words = 0

            # Calculate topic labels
            for word in ordered_words:
                if vocab_inv[word] != "<unk>" and vocab_inv[word] not in topic_labels:
                    topic_labels.append(vocab_inv[word])
                    break

            # Calculate most similar words and save them in word_df
            for word in ordered_words:
                if found_words == self.args.labeling_num_words:
                    break
                elif vocab_inv[word] != "<unk>":
                    word_df.loc[len(word_df)] = (
                        topic_labels[topic], found_words + 1, vocab_inv[word], str(round(sims[word], 2)))
                    found_words += 1

        return topic_labels, word_df

    # Returns a dataframe containing the most similar sentences for every topic
    def get_similar_sentences(self, topic_labels, topic_probs):
        """Selects the most similar sentences for every topic.

        Args:
            topic_labels: List that contains the topic labels
            topic_probs: Numpy array that contains the probability for every sentence-topic combination

        Returns:
            sentence_df: DataFrame that contains the most similar sentences for every topic

        """
        train_sen_file = codecs.open('./datasets/' + self.args.domain + '/train.txt', 'r', 'utf-8')
        sentences = []

        # Read in all sentences that are in the input data
        for line in train_sen_file:
            words = line.strip().split()
            sentences.append(words)

        # Calculate the sentences with the highest topic probabilities
        max_indices = np.argsort(topic_probs, axis=0)[::-1]
        max_probs = np.sort(topic_probs, axis=0)[::-1]

        sentence_df = pd.DataFrame(columns=['topic', 'rank', 'sentence'])
        similar_sentences = codecs.open(self.out_dir + '/similar_sentences', 'w', 'utf-8')

        # Iterate through the topics and get most similar sentences
        for topic_ind in range(self.args.num_topics):
            similar_sentences.write("Topic " + str(topic_ind) + ": " + str(topic_labels[topic_ind]) + "\n")
            curr_ind_col = max_indices[:, topic_ind]
            curr_prob_col = max_probs[:, topic_ind]

            # Write the most similar sentences to a file and save them to the sentence_df DataFrame
            for rank in range(self.args.num_sentences):
                similar_sentences.write(' '.join(sentences[curr_ind_col[rank]]) + " --> Probability: "
                                        + str(curr_prob_col[rank]) + "\n")
                sentence_df.loc[len(sentence_df)] = (
                    str(topic_labels[topic_ind]), rank + 1, ' '.join(sentences[curr_ind_col[rank]]))
            similar_sentences.write("\n")

        return sentence_df

    def get_json_objects(self, model, vocab_inv, topic_probs):
        """Retrieves the nodes and links that should be saved in the JSON file for the visualization.

        Args:
            model: Keras model object
            vocab_inv: Dictionary that maps the index of every word in the vocab file to the corresponding word (In descending order based on occurrences)
            topic_probs: Numpy array that contains the probability for every sentence-topic combination

        Returns:
            nodes: List that contains all the node objects that should be shown in the visualization
            links: List that contains all the link objects that should be shown in the visualization

        """

        topic_labels, word_df = self.get_similar_words(model, vocab_inv)
        sentences_df = self.get_similar_sentences(topic_labels, topic_probs)

        df = pd.DataFrame(topic_probs, columns=topic_labels)
        predict_labels = df.idxmax(axis=1)
        corr_df = df.corr(method="pearson")
        topic_occ = []

        # Calculate the topic occurrences
        for topic_label in topic_labels:
            topic_occ.append((predict_labels == topic_label).sum())

        nodes = []
        links = []

        # Specify the ranks for the most similar words and sentences based on the parameters
        top_word_ranks = [i for i in range(1, self.args.num_words + 1)]
        top_sen_ranks = [i for i in range(1, self.args.num_sentences + 1)]

        # Get the topic labels
        topic_labels = self.calculate_initial_labels(word_df)

        # Iterate through all topics and get the most similar words and sentences
        for i in range(corr_df.shape[1]):
            top_words = word_df[word_df["topic"] == str(corr_df.columns[i])].word[0:len(top_word_ranks)].values
            top_word_similarities = word_df[word_df["topic"] == str(corr_df.columns[i])].similarity[
                                    0:len(top_word_ranks)].values
            top_sentences = sentences_df[sentences_df["topic"] == str(corr_df.columns[i])].sentence.values

            word_objects = []
            sentence_objects = []

            # Create word and sentence objects and append them to the nodes and links lists
            for word_ind in range(len(top_words)):
                word_objects.append(
                    TopWord(top_word_ranks[word_ind], top_words[word_ind], top_word_similarities[word_ind]))

            for sen_ind in range(len(top_sentences)):
                sentence_objects.append(TopSentence(top_sen_ranks[sen_ind], top_sentences[sen_ind]))

            nodes.append(Node(str(topic_labels[i]), i, str(topic_occ[i]), word_objects, sentence_objects))
            for j in range(0, i):
                links.append(Link(nodes[i].id, nodes[j].id, corr_df.iloc[i, j].round(2)))

        return nodes, links

    def calculate_initial_labels(self, word_df):
        """Calculates the topic labels based on the number of shared hypernyms. If no shared hypernym is detected, the most similar word is used instead.

        Args:
            word_df: DataFrame that contains the most similar words of every topic

        Returns:
            topic_labels: List that contains the topic labels

        """

        topic_word_lists = []
        topic_labels = []
        curr_topic = 0
        num_hypernyms = 0
        hypernym_file = codecs.open(self.out_dir + '/topic_labels.log', 'w', 'utf-8')
        metric_file = codecs.open(self.out_dir + '/metrics.log', 'a', 'utf-8')
        metric_comparison_file = codecs.open('./code/output_dir/' + self.args.domain + '/metrics.log', 'a', 'utf-8')

        # Iterate through all the topics and append the most similar words
        for curr_ind in range(self.args.num_topics):
            topic_word_lists.append(
                word_df.iloc[curr_topic * self.args.labeling_num_words: self.args.labeling_num_words * (curr_topic + 1),
                2].values)
            curr_topic += 1

        # Go through the most similar words of every topic
        for topic_li in topic_word_lists:
            overall_hypernym_li = []
            path_distance = 0

            # Iterate through the words
            for word in topic_li:
                try:
                    inv_hypernym_path = wn.synsets(str(word))[0].hypernym_paths()[0][::-1]
                except:
                    continue
                specific_hypernym_li = []

                # Iterate through the hypernym path and only consider the path where distance <= distance to root hypernym
                for entry in inv_hypernym_path:
                    max_path_len = len(inv_hypernym_path) / 2

                    # Save hypernyms for every topic in a specific list
                    if path_distance < max_path_len:
                        specific_hypernym_li.append(str(entry)[8:-7])
                        path_distance += 1

                path_distance = 0

                # Save hypernyms of one topic in a large list that contains all hypernyms
                overall_hypernym_li.append(specific_hypernym_li)

            common_hypernyms = []

            # Index and index2 are the lists that contain the hypernyms for the given topic number (e.g. index=1 --> Hypernyms for topic 1)
            for index in range(len(overall_hypernym_li) - 1):
                for index2 in range(index + 1, len(overall_hypernym_li)):
                    hypernym_found = False
                    # Iterate over all hypernyms
                    for entry in overall_hypernym_li[index]:
                        for entry2 in overall_hypernym_li[index2]:
                            # Save the hypernym if two different words are compared and no lower hypernym was already found
                            if entry == entry2 and hypernym_found is False:
                                common_hypernyms.append(entry)
                                hypernym_found = True
                                break
                            else:
                                continue

            # If no hypernyms are found, use the most similar word
            if len(common_hypernyms) == 0:
                top_word = self.get_top_word(topic_li, topic_labels)

            # If hypernyms are found, get the hypernym with the lowest number of occurrences that is not already used
            else:
                top_word = self.get_top_hypernym(topic_li, topic_labels, Counter(common_hypernyms).most_common())
                num_hypernyms += sum(Counter(common_hypernyms).values())

            topic_labels.append(top_word)
            hypernym_file.write('Topic %s:' % (top_word) + "\n")
            hypernym_file.write('   - Common hypernyms: %s' % (Counter(common_hypernyms).most_common()) + "\n")
            hypernym_file.write('   - Similar words: %s' % (topic_li) + "\n" + "\n")

        # Write information to multiple logging files
        avg_num_hypernyms = float("{0:.2f}".format(num_hypernyms / float(self.args.num_topics)))
        hypernyms_per_word = float("{0:.2f}".format(avg_num_hypernyms / float(self.args.labeling_num_words)))

        hypernym_file.write('Hypernyms per Word: %s' % (hypernyms_per_word) + "\n")
        hypernym_file.write('Average number of hypernyms: %s' % (avg_num_hypernyms) + "\n")
        hypernym_file.write('Number of hypernyms found: %s' % num_hypernyms + "\n")

        metric_file.write('Hypernyms per Word: %s' % (hypernyms_per_word) + "\n")
        metric_file.write('Average number of hypernyms: %s' % (avg_num_hypernyms) + "\n")
        metric_file.write('Number of hypernyms found: %s' % num_hypernyms + "\n" + "\n")

        metric_comparison_file.write('Hypernyms per Word: %s' % (hypernyms_per_word) + "\n")
        metric_comparison_file.write('Average number of hypernyms: %s' % (avg_num_hypernyms) + "\n")
        metric_comparison_file.write('Number of hypernyms found: %s' % num_hypernyms + "\n" + "\n")

        return topic_labels

    def get_top_word(self, topic_li, topic_labels):
        """Retrieves the most similar word and sets it as label.

        Args:
            topic_li: Numpy array that contains the most similar words
            topic_labels: List that contains the previous topic labels (Required because we do not want duplicate topic labels)

        Returns:

        """
        # Iterate through most similar words and take first one that is not already used and unequal to <unk>
        for word in topic_li:
            if word != "<unk>" and word not in topic_labels:
                return word

        # If every shared hypernyms is already used as label and every similar word is already used, use the generic name "topic_" + index instead
        return "topic_" + str(len(topic_labels))

    def get_top_hypernym(self, topic_li, topic_labels, common_hypernyms):
        """Retrives the most commonly shared lowest hypernym.

        Args:
            topic_li: Numpy array that contains the most similar words
            topic_labels: List that contains the previous topic labels (Required because we do not want duplicate topic labels)
            common_hypernyms: List that contains the shared hypernyms as (entry, occurrence) tuples

        Returns:

        """
        # Iterate through the common hypernyms and use the most frequent one that is not already used as label
        for common_hypernym in common_hypernyms:
            if common_hypernym[0] not in topic_labels:
                return common_hypernym[0]

        # If all shared hypernyms are already used as label for another topic, use the top word instead
        return self.get_top_word(topic_li, topic_labels)

    def write_json(self, model, vocab_inv, topic_probs):
        """Writes all relevant topic information to a JSON file so that it can be imported in the visualization and labeling tool.

        Args:
            model: Keras model object
            vocab_inv: Dictionary that maps the index of every word in the vocab file to the corresponding word (In descending order based on occurrences)
            topic_probs: Numpy array that contains the probability for every sentence-topic combination

        Returns:

        """

        nodes, links = self.get_json_objects(model, vocab_inv, topic_probs)
        self.logger.info('Writing .json file...')

        # Create a String that contains all the information in a .json format
        node_str = '{ "nodes": ['
        link_str = ' "links": ['

        for node in nodes[:-1]:
            node_str += node.to_json() + ","

        node_str += nodes[-1].to_json() + " ],"

        for link in links[:-1]:
            link_str += link.to_json() + ","

        link_str += links[-1].to_json() + " ] }"

        json_str = node_str + link_str

        with open(self.vis_path + "/topic_information.json", "w") as f:
            f.write(json_str)

        self.logger.info('.json written successfully')

    def build_model(self):
        """Creates the model object, which is used to calculate topics, similar words, similar sentences, topic occurrences, and topic similarities

        Returns:
            model: Keras model object

        """
        optimizer = get_optimizer(self.args)
        self.logger.info('Building model')

        self.logger.info('   Number of training examples: %d', len(self.train_x))
        self.logger.info('   Length of vocab: %d', len(self.vocab))

        def max_margin_loss(y_true, y_pred):
            return K.mean(y_pred)

        model = create_model(self.args, self.overall_maxlen, self.vocab)

        # Freeze the word embedding layer
        model.get_layer('word_emb').trainable = False

        # Check option to fix clusters instead of training them
        if self.args.fix_clusters == "yes":
            model.get_layer('topic_emb').trainable = False

        model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])

        return model

    def train_model(self, model):
        """Train the model based on the hyperparameters defined.

        Args:
            model: Keras model object that is returned after calling Train.build_model()

        Returns:

        """

        vocab_inv = {}
        for w, ind in self.vocab.items():
            vocab_inv[ind] = w

        sen_gen = self.sentence_batch_generator(self.train_x, self.args.batch_size)
        neg_gen = self.negative_batch_generator(self.train_x, self.args.batch_size, self.args.neg_size)

        batches_per_epoch = len(self.train_x) / self.args.batch_size
        # batches_per_epoch = 1000

        self.logger.info("Batches per epoch: %d", batches_per_epoch)
        self.logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')
        min_loss = float('inf')
        loss_li = []

        for ii in xrange(self.args.epochs):
            t0 = time()
            loss, max_margin_loss = 0., 0.

            for b in tqdm(xrange(batches_per_epoch)):
                sen_input = sen_gen.next()
                neg_input = neg_gen.next()

                batch_loss, batch_max_margin_loss = model.train_on_batch([sen_input, neg_input],
                                                                         np.ones((self.args.batch_size, 1)))

                loss += batch_loss / batches_per_epoch
                max_margin_loss += batch_max_margin_loss / batches_per_epoch

            tr_time = time() - t0

            self.logger.info('Epoch %d, train: %is' % (ii + 1, tr_time))
            self.logger.info('   Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (
                loss, max_margin_loss, loss - max_margin_loss))

            if loss < min_loss:
                self.logger.info('   Loss < min_loss')

                min_loss = loss
                word_emb = K.get_value(model.get_layer('word_emb').embeddings)
                topic_emb = K.get_value(model.get_layer('topic_emb').W)
                word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
                topic_emb = topic_emb / np.linalg.norm(topic_emb, axis=-1, keepdims=True)

                model.save_weights(self.out_dir + '/model_param')
                self.write_topics(word_emb, topic_emb, ii + 1, vocab_inv)

                training_detail_file = codecs.open(self.out_dir + '/training_details.log', 'a', 'utf-8')
                training_detail_file.write('Epoch %d, train: %is' % (ii + 1, tr_time) + "\n")
                training_detail_file.write('Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (
                    loss, max_margin_loss, loss - max_margin_loss) + "\n")
                loss_li.append(float("{0:.4f}".format(loss)))

            else:
                self.logger.info('   Loss > min_loss')
                loss_li.append(float("{0:.4f}".format(loss)))

            # In Final Epoch
            if ii + 1 == self.args.epochs:
                self.logger.info('Training finished')
                self.logger.info('Calculating most representative topic sentences...')
                test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])


                # If argument is not given explicitly by the user calculate good default value (One batch per 5000 entries)
                if self.args.probability_batches == 0:
                    num_probability_batches = len(self.train_x) / 5000
                    self.logger.info('Using %s probability batches...', num_probability_batches)

                else:
                    num_probability_batches = self.args.probability_batches

                split_inputs = np.array_split(self.train_x, num_probability_batches)

                _, topic_probs = test_fn([split_inputs[0], 0])

                for split_input in split_inputs[1:]:
                    _, curr_topic_prob = test_fn([split_input, 0])
                    topic_probs = np.append(topic_probs, curr_topic_prob, axis=0)

                self.logger.info('Most representative sentences calculated successfully')
                self.write_json(model, vocab_inv, topic_probs)
                self.save_model_loss(loss_li)
                # os.system(
                #    "python ./code/coherence_score.py -f ./code/output_dir/organic_food_preprocessed/" + self.args.conf + "/topics.log -c ./preprocessed_data/organic_food_preprocessed/train.txt -o ./code/output_dir/organic_food_preprocessed/" + self.args.conf)

    def save_model_loss(self, loss_li):
        """Creates plots of the training loss and saves them as .png and .pdf files.

        Args:
            loss_li: List that contains the model loss for every epoch

        Returns:

        """
        metric_file = codecs.open(self.out_dir + '/metrics.log', 'a', 'utf-8')
        metric_comparison_file = codecs.open('./code/output_dir/' + self.args.domain + '/metrics.log', 'a', 'utf-8')
        metric_file.write('Final loss: %s' % (loss_li[-1]) + "\n")
        metric_file.write('Loss development: %s' % (loss_li) + "\n" + "\n")
        metric_comparison_file.write('Final loss: %s' % (loss_li[-1]) + "\n")
        metric_comparison_file.write('Loss development: %s' % (loss_li) + "\n" + "\n")

        epoch_li = [epoch for epoch in range(1, self.args.epochs + 1)]
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlabel("Epoch", fontsize=18, weight="bold")
        ax.set_ylabel("Loss", fontsize=18, weight="bold")
        ax.set_title('Model loss', fontsize=20, weight="bold")
        plt.plot(epoch_li, loss_li)
        plt.savefig(self.out_dir + "/model_loss.pdf", format="pdf")
        plt.savefig(self.out_dir + "/model_loss.png", format="png")


def main():
    logging.basicConfig(
        filename='out.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', required=True,
                        help="domain of the corpus")
    parser.add_argument("--conf", dest="conf", type=str, metavar='<str>', required=True,
                        help="Train configuration for the given domain")
    parser.add_argument("--emb-path", dest="emb_path", type=str, metavar='<str>', required=True,
                        help="The path to the word embedding file")
    parser.add_argument("--num-topics", dest="num_topics", type=int, metavar='<int>', default=20,
                        help="The number of topics specified that are calculated by the model (default=20)")
    parser.add_argument("--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000,
                        help="Vocab size. '0' means no limit (default=9000)")
    parser.add_argument("--num-words", dest="num_words", type=int, metavar='<int>', default=10,
                        help="Number of most similar words displayed for each topic")
    parser.add_argument("--num-sentences", dest="num_sentences", type=int, metavar='<int>', default=10,
                        help="Number of most similar sentences displayed for each topic")
    parser.add_argument("--labeling-num-words", dest="labeling_num_words", type=int, metavar='<int>', default=25,
                        help="Number of most similar words used to generate the labels")
    parser.add_argument("--batch-size", dest="batch_size", type=int, metavar='<int>', default=64,
                        help="Batch size used for training (default=64)")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=20,
                        help="Number of epochs (default=20)")
    parser.add_argument("--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                        help="Number of negative instances (default=20)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                        help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument("--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                        help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
    parser.add_argument("--fix-clusters", dest="fix_clusters", type=str, metavar='<str>', default="no",
                        help="Fix initial clusters (yes or no)")
    parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                        help="The weight of orthogonal regularization (default=0.1)")
    parser.add_argument("--probability-batches", dest="probability_batches", type=int, metavar='<int>', default=0,
                        help="Calculation of topic probabilities is split into batches to avoid out of memory error."
                             "If an out of memory error or bus error occurs, increase this value.")
    parser.add_argument("--emb-dim", dest="emb_dim", type=int, metavar='<int>', default=300,
                        help="Embeddings dimension (default=300)")
    parser.add_argument("--emb-type", dest="emb_type", type=str, metavar='<str>', default="glove_finetuned",
                        help="The type of word vectors to use (glove, both, w2v, word2vec_finetune, glove_finetuned)")









    args = parser.parse_args()
    out_dir = './code/output_dir/' + args.domain + '/' + args.conf
    U.mkdir_p(out_dir)
    U.mkdir_p(out_dir + "/topics")
    U.print_args(args, out_dir + '/train_params')
    U.print_args(args, out_dir + '/metrics.log')

    assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}, "Invalid algorithm argument"
    assert args.fix_clusters in {'yes', 'no'}, "Invalid fix_clusters argument"
    assert args.labeling_num_words >= args.num_words, "Number of words used to generate labels must be >= Number of words displayed in visualization"

    np.random.seed(1234)

    trainer = Train(args, logger, out_dir)
    model = trainer.build_model()
    trainer.train_model(model)


if __name__ == "__main__":
    main()

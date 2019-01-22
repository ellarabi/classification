import os
import sys
import codecs
import collections

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.cluster import KMeans

from nltk import tokenize
from nltk import word_tokenize

import numpy as np
import time, math
import pickle

import en_cohesive_markers
import en_function_words
import en_word_ranks


class Utils:
    @staticmethod
    def load_words_list(filename):
        """
		loading word list from palin text file
		:param filename: filename to load
		:return: the list of words
		"""
        with codecs.open(filename, 'r', 'utf-8') as fin:
            words = [line.split()[0] for line in fin]
        # end with
        return words

    # end def

    @staticmethod
    def zipngram(text, n):
        """
		generates ngrams from a text
		:param text: the text subject to ngrams
		:param n: a gram size
		:return: ngrams
		"""
        return zip(*[text.split()[i:] for i in range(n)])

    # end def

    @staticmethod
    def zipCharNgram(text, n):
        """
		generates ngrams from a text
		:param text: the text subject to ngrams
		:param n: a gram size
		:return: ngrams
		"""
        return zip(*[text[i:] for i in range(n)])

    # end def

    @staticmethod
    def common_words(filename, threshold_max, threshold_min):
        """
		returns a list of words from a file with # of occurrences in a range
		:param filename: filename to read words and their occurrences
		:param threshold_max: upper bound
		:param threshold_min: lower bound
		:return: list of words
		"""
        common = []
        with open(filename, 'r') as fin:
            for line in fin:
                frequency = line.strip().split()[1]
                if int(frequency) > threshold_max or int(frequency) < threshold_min: continue

                token = line.strip().split()[0]
                common.append(token)
            # end for
        # end with
        return common

    # end def

    @staticmethod
    def parse_classification_configuration(cfg_filename):
        configuration = []
        with open(cfg_filename, 'r') as fin:
            for line in fin:
                if (line.startswith("#") or not line.strip()):
                    continue
                # end if
                configuration.append(label(line.split()[0], line.split()[1], line.split()[2]))
            # print('appended', line.strip())
            # end for
        # end with
        return configuration

    # end def

    @staticmethod
    def divide_into_chunks(configuration):
        if CHUNK_MODE == 'sent':
            return Utils.divide_into_sentences(configuration)
        else:
            return Utils.divide_into_tokens(configuration)

    # end if
    # end def

    @staticmethod
    def divide_into_sentences(configuration):
        labels = []
        text_chunks = []
        for entry in configuration:
            with codecs.open(entry.datafile, 'r', ENCODING) as fin:
                processed_chunks = 0
                for line in fin:
                    text_chunks.append(line.strip())
                    labels.append(entry.name)
                    processed_chunks += 1

                    if processed_chunks == int(entry.chunks):
                        break
                    # end if
                # end for
            # end with
            print("loaded", entry.chunks, "sentences from", entry.datafile, processed_chunks)
        # end for
        return text_chunks, labels

    # end def

    @staticmethod
    def divide_into_tokens(configuration):
        labels = []
        text_chunks = []
        for entry in configuration:
            print("loading", entry.chunks, "chunks from", entry.datafile)
            with codecs.open(entry.datafile, 'r', ENCODING) as fin:
                text = fin.read().strip()
                tokens = text.split()

                processed_chunks = 0
                for i in range(0, len(tokens), CHUNK_SIZE):
                    text_chunks.append(' '.join(tokens[i:i + CHUNK_SIZE]).lower())
                    labels.append(entry.name)
                    processed_chunks += 1

                    if processed_chunks == int(entry.chunks):
                        break
                    # end if
                # end for
            # end with
        # end for
        return text_chunks, labels

    # end def

    @staticmethod
    def print_chunks_data(text_chunks, labels):
        print('total', len(text_chunks), 'for classification')
    # end def

    @staticmethod
    def filter_out(ngram):
        should_filter_out = False
        for unigram in ngram:
            if not unigram.isalpha():
                should_filter_out = True
                break
            # end if
        # end for
        return should_filter_out

    # end def

    @staticmethod
    def argmax(vec):
        return np.argmax(vec)
    # end def

# end class


class MathUtils:
    @staticmethod
    def normalize(feature_values):
        """
		feature normalization by column towards classification
		:param feature_values:
		:return:
		"""
        scaler = preprocessing.MaxAbsScaler().fit(feature_values)
        n_features = scaler.transform(feature_values)
        return n_features

# end class


class Classification:

    def __cluster(self, text_quantified, instances, labels):
        norm_text_quantified = MathUtils.normalize(text_quantified)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(norm_text_quantified)
        return list(kmeans.labels_)

    # end def

    def __classify(self, text_quantified, instances, labels):
        import warnings
        warnings.filterwarnings("ignore")

        # define 10-fold cross-validation with data shuffling
        kf = cross_validation.KFold(instances, n_folds=10, shuffle=True)

        # classify with logistic regression
        norm_text_quantified = MathUtils.normalize(text_quantified)
        clf = LogisticRegression().fit(norm_text_quantified, labels)  # classification
        scores = cross_validation.cross_val_score(clf, norm_text_quantified, labels, cv=kf)
        self._check_confusion(clf, norm_text_quantified, labels, kf)

        return scores.mean()

    # end def

    def _check_confusion(self, clf, norm_text_quantified, labels, kf):
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import cross_val_predict
        predicted = cross_val_predict(clf, norm_text_quantified, labels, cv=kf)
        conf_mat = confusion_matrix(labels, predicted)
        print(conf_mat)

    # end def

    def __train_and_test(self, train_features, train_labels, test_features, test_labels):
        norm_text_quantified = MathUtils.normalize(train_features)
        clf = LogisticRegression().fit(norm_text_quantified, train_labels)
        predicted = clf.predict(MathUtils.normalize(test_features))

        predicted_probs = clf.predict_proba(MathUtils.normalize(test_features))

        with open('classification.out.tsv', 'w') as fout:
            for i, (plabel, prob) in enumerate(zip(predicted, predicted_probs)):
                fout.write(str(i) + '\t' + plabel + '\t' + '{0:.3f}'.format(np.max(prob)) + '\n')
            # end for
        # end with
    # end def

    def __extract_features(self, configuration, text_chunks):
        labels = np.array(['dummy'])
        values = np.zeros((len(text_chunks), 1))

        # values, labels = self.__add_feature(self.__extract_function_words(text_chunks,
        #    en_function_words.FUNCTION_WORDS), values, labels)
        # values, labels = self.__add_feature(self.__extract_cohesive_markers(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_pos_tags(configuration), values, labels)

        # values, labels = self.__add_feature(self.__extract_type_to_token_ratio_v1(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_type_to_token_ratio_v2(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_type_to_token_ratio_v3(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_mean_token_length(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_mean_sentence_length(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_most_frequent_words(text_chunks, 5), values, labels)
        # values, labels = self.__add_feature(self.__extract_most_frequent_words(text_chunks, 10), values, labels)
        # values, labels = self.__add_feature(self.__extract_most_frequent_words(text_chunks, 2000), values, labels)
        # values, labels = self.__add_feature(self.__extract_mean_word_rank_v1(text_chunks), values, labels)
        # values, labels = self.__add_feature(self.__extract_mean_word_rank_v2(text_chunks), values, labels)

        values, labels = self.__add_feature(self.__extract_token_ngrams(text_chunks, 1), values, labels)
        values, labels = self.__add_feature(self.__extract_token_ngrams(text_chunks, 2), values, labels)
        values, labels = self.__add_feature(self.__extract_token_ngrams(text_chunks, 3), values, labels)

        assert (values.shape[1] == len(labels))
        print('instances and features:', values.shape)

        return values, labels

    # end def

    def __add_feature(self, results, values, labels):
        values = np.hstack((values, np.matrix(results[0])))
        labels = np.hstack((labels, np.array(results[1])))
        print('features size:', values.shape)

        return values, labels

    # end def

    def __selectKbest(self, feature_values, labels, feature_names):
        assert (feature_values.shape[0] == len(labels))
        trans = SelectKBest(k=25).fit(feature_values, labels)

        best_feature_names = []
        for i in trans.get_support(indices=True):
            best_feature_names.append(feature_names[i])
        # end for

        return best_feature_names

    # end def

    def __extract_top_k_token_ngrams(self, text, n):
        print('extracting top-k token ngrams...')
        token_ngrams_dict = {}
        for ngram in Utils.zipngram(text.lower(), n):
            # if Utils.filter_out(ngram): continue

            count = token_ngrams_dict.get(ngram, 0)
            token_ngrams_dict[ngram] = count + 1
        # end for

        return sorted(token_ngrams_dict, key=token_ngrams_dict.__getitem__, reverse=True)[:K]

    # end def

    def __extract_token_ngrams(self, instances, n):

        print('extracting token ngrams...')
        top_k_token_ngrams = self.__extract_top_k_token_ngrams(' '.join(instances), n)

        feature_values = []
        for instance in instances:
            instance_values = []
            token_ngram_2_count = {}
            for ngram in Utils.zipngram(instance.lower(), n):
                count = token_ngram_2_count.get(ngram, 0)
                token_ngram_2_count[ngram] = count + 1
            # end for

            for top_token_ngram in top_k_token_ngrams:
                instance_values.append(token_ngram_2_count.get(top_token_ngram, 0))
            # end for

            feature_values.append(instance_values)
        # end for

        top_k_token_ngrams_flat = [' '.join(token_ngram) for token_ngram in top_k_token_ngrams]
        return feature_values, top_k_token_ngrams_flat

    # end def

    def __extract_top_k_char_ngrams(self, text, n):
        print('extracting top-k char ngrams...')
        char_ngrams_dict = {}
        for ngram in Utils.zipCharNgram(text.lower(), n):
            count = char_ngrams_dict.get(ngram, 0)
            char_ngrams_dict[ngram] = count + 1
        # end for

        return sorted(char_ngrams_dict, key=char_ngrams_dict.__getitem__, reverse=True)[:K]

    # end def

    def __extract_char_ngrams(self, instances, n):

        print('extracting char ngrams...')
        top_k_char_ngrams = self.__extract_top_k_char_ngrams(' '.join(instances), n)

        feature_values = []
        for instance in instances:
            instance_values = []
            char_ngram_2_count = {}
            for ngram in Utils.zipCharNgram(instance.lower(), n):
                count = char_ngram_2_count.get(ngram, 0)
                char_ngram_2_count[ngram] = count + 1
            # end for

            for top_char_ngram in top_k_char_ngrams:
                instance_values.append(char_ngram_2_count.get(top_char_ngram, 0))
            # end for

            feature_values.append(instance_values)
        # end for

        top_k_char_ngrams_flat = [' '.join(char_ngram) for char_ngram in top_k_char_ngrams]
        return feature_values, top_k_char_ngrams_flat

    # end def

    def __extract_function_words(self, instances, function_words):
        feature_values = []
        for i, instance in enumerate(instances):
            instance_values = []
            split_instance = instance.split()
            for fw in function_words:
                instance_values.append(split_instance.count(fw))
            # end if
            assert (len(instance_values) == len(function_words))
            feature_values.append(instance_values)
        # end for

        print('finished extracting function words')
        return feature_values, function_words

    # end def

    def __extract_type_to_token_ratio_v1(self, instances):
        feature_values = []
        for instance in instances:
            ttr = float(len(set(instance.split()))) / len(instance.split())
            feature_values.append(ttr)
        # end for

        return np.matrix(feature_values).transpose(), ['type-to-token-ratio_v1']

    # end def

    def __extract_type_to_token_ratio_v2(self, instances):
        feature_values = []
        for instance in instances:
            ttr = float(math.log(len(set(instance.split())))) / math.log(len(instance.split()))
            feature_values.append(ttr)
        # end for
        return np.matrix(feature_values).transpose(), ['type-to-token-ratio_v2']

    # end def

    def __extract_type_to_token_ratio_v3(self, instances):
        feature_values = []
        for instance in instances:
            words = collections.Counter()
            words.update(instance.split())
            single_appearance = 0
            for word in list(words.items()):
                if word[1] == 1:
                    single_appearance += 1
                # end if
            # end for

            N = len(instance.split())
            V = len(set(instance.split()))
            ttr = float(100 * math.log(N)) / (1 - float(single_appearance) / V)
            feature_values.append(ttr)
        # end for
        return np.matrix(feature_values).transpose(), ['type-to-token-ratio_v3']

    # end def

    def __extract_mean_sentence_length(self, instances):
        feature_values = []
        for instance in instances:
            considered = 0
            total_sentences_length = 0
            sentences = tokenize.sent_tokenize(instance)
            for sentence in sentences:
                if (len(sentence.split())) > 500:
                    continue
                # end if

                total_sentences_length += len(sentence.split())
                considered += 1
            # end for

            feature_values.append(float(total_sentences_length) / considered)
        # end for
        return np.matrix(feature_values).transpose(), ['mean-sentence-length']

    # end def

    def __extract_mean_token_length(self, instances):
        feature_values = []
        for instance in instances:
            total_token_length = 0
            for token in instance.split():
                total_token_length += len(token)
            # end for

            feature_values.append(float(total_token_length) / len(instance.split()))
            # print(len(tokenize.sent_tokenize(instance)))
        # end for
        return np.matrix(feature_values).transpose(), ['mean-token-length']

    # end def

    def __extract_mean_word_rank_v1(self, instances):
        feature_values = []
        for instance in instances:
            total_word_ranks = 0
            for token in instance.split():
                # print(en_word_ranks.WORD_RANKS.get(token.lower(), 6000))
                total_word_ranks += en_word_ranks.WORD_RANKS.get(token.lower(), 6000)
            # end for

            feature_values.append(float(total_word_ranks) / len(instance.split()))
        # end for
        return np.matrix(feature_values).transpose(), ['mean-word_rank_v1']

    # end def

    def __extract_mean_word_rank_v2(self, instances):
        feature_values = []
        for instance in instances:
            total_word_ranks = 0
            for token in instance.split():
                # print(en_word_ranks.WORD_RANKS.get(token.lower(), 0))
                total_word_ranks += en_word_ranks.WORD_RANKS.get(token.lower(), 0)
            # end for

            feature_values.append(float(total_word_ranks) / len(instance.split()))
        # end for
        return np.matrix(feature_values).transpose(), ['mean-word_rank_v2']

    # end def

    def __extract_most_frequent_words(self, instances, k):
        # high_ranking_words = [word for (word, rank) in en_word_ranks.WORD_RANKS.items() if rank <= k]
        high_ranking_words = [word for (word, rank) in en_word_ranks.WORD_RANKS.items() if rank <= k and
                              word not in en_function_words.FUNCTION_WORDS]

        top_words = sorted(high_ranking_words, key=lambda word: en_word_ranks.WORD_RANKS[word])

        feature_values = []
        for word in top_words:
            word_values = []
            for instance in instances:
                word_values.append(instance.split().count(word))
            # end if
            feature_values.append(word_values)
            print('finished', word)
        # end for

        assert (len(feature_values) == len(top_words))
        return np.matrix(feature_values).transpose(), top_words

    # end def

    def __extract_cohesive_markers(self, instances):
        feature_values = []
        tokenized_markers = [(marker, word_tokenize(marker)) for marker in en_cohesive_markers.COHESIVE_MARKERS]

        for j, instance in enumerate(instances):
            instance_values = []
            split_instance = instance.split()
            for marker, tokenized in tokenized_markers:
                count = 0
                for i, _ in enumerate(split_instance):
                    if tokenized == split_instance[i:i + len(tokenized)]:
                        count += 1
                # end if
                # end for
                instance_values.append(count)
            # end for
            feature_values.append(instance_values)
            # print('finished with instance', j)
        # end for

        print('finished extracting cohesive markers')
        return feature_values, en_cohesive_markers.COHESIVE_MARKERS

    # end def

    def __extract_pos_tags(self, configuration):
        configuration_pos = []

        for entry in configuration:
            configuration_pos.append(label(entry.datafile + '.pos', entry.name, entry.chunks))
        # end for

        dirname = '/'.join(configuration[0].datafile.split('/')[:-1])
        print('generated pos_tagged configuration, assuming pos texts in', dirname)

        text_chunks, labels = Utils.divide_into_chunks(configuration_pos)
        print('divided pos_tagged texts into', len(text_chunks), 'chunks')

        return self.__do_extract_pos_tags(text_chunks)

    # end def

    def __do_extract_pos_tags(self, instances):
        instances_pos = []
        for instance in instances:
            current_instance_pos = []
            tokens_with_pos = instance.split()
            for token in tokens_with_pos:
                try:
                    current_instance_pos.append(token.split('_')[1])
                except:
                    current_instance_pos.append('UNK')
            # end try
            # end for

            # append current chunk pos tags to tags array
            instances_pos.append(' '.join(current_instance_pos))
        # end for

        print('total instances with pos tags:', len(instances_pos))
        top_k_pos_ngrams = self.__extract_top_k_pos_tags(' '.join(instances_pos))
        # Serialization.save_obj(top_k_pos_ngrams, 'pos.ngrams.top.3K')
        # top_k_pos_ngrams = Serialization.load_obj('pos.ngrams.top.3K')[:K]
        # print(top_k_pos_ngrams)

        feature_values = []
        for instance in instances_pos:
            instance_values = []
            pos_ngram_2_count = {}
            for ngram in Utils.zipngram(instance.upper(), N):
                count = pos_ngram_2_count.get(ngram, 0)
                pos_ngram_2_count[ngram] = count + 1
            # end for

            for top_pos_ngram in top_k_pos_ngrams:
                instance_values.append(pos_ngram_2_count.get(top_pos_ngram, 0))
            # end for

            feature_values.append(instance_values)
        # end for

        top_k_pos_ngrams_flat = [' '.join(pos_ngram) for pos_ngram in top_k_pos_ngrams]
        print('finished extracting pos-ngrams')

        return feature_values, top_k_pos_ngrams_flat

    # end def

    def __extract_top_k_pos_tags(self, tagged_text):
        pos_ngrams_dict = {}
        for ngram in Utils.zipngram(tagged_text.upper(), N):
            count = pos_ngrams_dict.get(ngram, 0)
            pos_ngrams_dict[ngram] = count + 1
        # end for

        return sorted(pos_ngrams_dict, key=pos_ngrams_dict.__getitem__, reverse=True)[:K]

    # end def

    def cross_validate(self, cfg_filename):
        """
		the entire process of loading data, chunking, feature extraction and classification
		:param cfg_filename: file where classification configuration is kept
		:return:
		"""
        start = time.clock()
        configuration = Utils.parse_classification_configuration(cfg_filename)
        text_chunks, labels = Utils.divide_into_chunks(configuration)

        # extract features
        feature_values, feature_names = self.__extract_features(configuration, text_chunks)

        # perform classification
        # print('len(labels):', len(labels))
        accuracy = self.__classify(feature_values, len(feature_values), labels)
        print('classification accuracy:', '{0:.7f}'.format(accuracy))

        # select k best features
        k_best_feature_names = self.__selectKbest(feature_values, labels, feature_names)
        print(k_best_feature_names)

        print('time:', '{0:.3f}'.format(time.clock() - start))

    # end def

    def train_predict(self, cfg_filename_train, cfg_filename_test):
        """
		train on some data and predict on another
		:param cfg_filename_train: configuration for training
		:param cfg_filename_test: configuration for test
		:return:
		"""
        start = time.clock()
        configuration_train = Utils.parse_classification_configuration(cfg_filename_train)
        train_chunks, train_labels = Utils.divide_into_chunks(configuration_train)

        # extract features
        train_feature_values, train_feature_names = self.__extract_features(configuration_train, train_chunks)
        print('extracted features from train set')

        configuration_test = Utils.parse_classification_configuration(cfg_filename_test)
        test_chunks, test_labels = Utils.divide_into_chunks(configuration_test)

        # extract features
        test_feature_values, test_feature_names = self.__extract_features(configuration_test, test_chunks)
        print('extracted features from test set')

        self.__train_and_test(train_feature_values, train_labels, test_feature_values, test_labels)

        print('time:', '{0:.3f}'.format(time.clock() - start))

    # end def

    def create_features_csv(self, cfg_filename, filename):
        """
		extract features and record results into csv format
		:param cfg_filename: configuration for feature extraction
		:param filename: file to dump the results
		:return:
		"""
        start = time.clock()
        configuration = Utils.parse_classification_configuration(cfg_filename)
        text_chunks, labels = Utils.divide_into_chunks(configuration)

        # extract features
        feature_values, feature_names = self.__extract_features(configuration, text_chunks)

        np.savetxt(filename, feature_values, delimiter=",")
        # np.savetxt("reddit.etymology.names.csv",  feature_names,  delimiter=",")
        # print('\n'.join(feature_names))

        print('time:', '{0:.3f}'.format(time.clock() - start))

    # end def

    def classify_bert(self, dirname):
        train_pos = dirname + 'bert.output.val.csv'
        train_neg = dirname + 'bert.output.inv.csv'
        test_org = dirname + 'bert.output.test.org.csv'
        test_alt = dirname + 'bert.output.test.alt.csv'
        self._train_predict_bert(train_pos, train_neg, test_org, test_alt)

    # end def

    def _train_predict_bert(self, pos_csv, neg_csv, test_org_csv, test_alt_csv):
        """
		train on pre-extracted features and predict on others
		:param pos_csv: features for positive class
		:param neg_csv: features for negative class
		:return:
		"""
        start = time.clock()
        pos_features = np.loadtxt(pos_csv, delimiter=",", skiprows=0)
        pos_labels = ['val'] * len(pos_features)
        neg_features = np.loadtxt(neg_csv, delimiter=",", skiprows=0)
        neg_labels = ['inv'] * len(neg_features)

        feature_values = pos_features[:]
        feature_values = np.vstack((feature_values, neg_features))
        print('features size:', feature_values.shape)

        labels = pos_labels[:]
        labels.extend(neg_labels)
        print('labels size:', len(labels))

        print('extracted features from train set, 10-fold classification...')
        accuracy = self.__classify(feature_values, len(feature_values), labels)
        print('10-fold classification accuracy:', '{0:.7f}'.format(accuracy))

        print('loading test set (original and alternative)...')
        org_features = np.loadtxt(test_org_csv, delimiter=",", skiprows=0)
        alt_features = np.loadtxt(test_alt_csv, delimiter=",", skiprows=0)

        print('predicting on test set...')
        self.__train_and_predict_bert(feature_values, labels, org_features, alt_features)

        print('time:', '{0:.3f}'.format(time.clock() - start))

    # end def

    def generate_features_dict(self, countries, feature_names, feature_values):
        dictionary = {}
        for i, country in enumerate(countries):
            dictionary[country] = {}  # all feature values for a country
            current_values = np.array(feature_values[i, :].tolist()[0])  # relevant matrix row

            assert (len(current_values) == len(feature_names))
            for feature_name, feature_value in zip(feature_names, current_values):
                dictionary[country][feature_name] = feature_value
            # end for
        # end for
        return dictionary
    # end def

# end class


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
		serialization of an object
		:param obj: object to serialize
		:param name: file name to store the object
		"""
        with open('pickle/' + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)

        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
		de-serialization of an object
		:param name: file name to load the object from
		"""
        with open('pickle/' + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)
        # end with
    # end def

# end class


ENCODING = 'utf-8'
CHUNK_MODE = 'sent'  # the other option is token
CHUNK_SIZE = 500  # (174, 870, 1740) 2500000 # 2000 # the max amount we have in tokens
label = collections.namedtuple('label', ['datafile', 'name', 'chunks'])
K = 300  # top-k
N = 3

if __name__ == '__main__':
    cl = Classification()

    # invocation: python classification.py input.data.cfg
    cl.cross_validate(sys.argv[1])  # standard evaluation and select k-best
    # cl.train_predict(sys.argv[1], sys.argv[2])

# end if

import re
from collections import defaultdict, namedtuple
from copy import deepcopy

from scipy.sparse import lil_matrix
from sklearn.linear_model import LogisticRegression


class MemmViterbi:
    _prefix_max_length = _suffix_max_length = 4

    _not_unknown_count_threshold = 3

    _upper_case_with_digit_and_dash_re = re.compile('[-A-Z0-9]+')

    def train(self, annotated_sentences):
        """ trains the MEMM model based on various word-features.
        annotated_sentences: a list of sentences where each sentence is a list of tagged words, where each tagged word
        is a tuple of (<word>, <TAG>), e.g., ('Greece', 'PROPN'),
        """
        self._init_word_data(annotated_sentences)
        # noinspection PyPep8Naming
        X = self._feature_vectors_from_annotated_sentences(annotated_sentences)
        y = self._target_vector_from_annotated_sentences(annotated_sentences)
        # noinspection PyAttributeOutsideInit
        self._lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, n_jobs=-1)
        self._lr.fit(X, y)

        return self

    def predict(self, sentence):
        """
        Predicts the sequence of the given sentence using the trained MEMM model in the Viterbi algorithm
        :param sentence: given as a list of words
        :return: a list of tags
        """
        probabilities = self._get_probabilities(sentence)
        viterbi = self._get_viterbi_matrix(probabilities, sentence)
        prediction = []
        max_prob = max(cell.prob for cell in viterbi[-1].values())
        previous = None
        for st, data in viterbi[-1].items():
            if data.prob == max_prob:
                prediction.append(st)
                previous = st
                break
        for t in range(len(viterbi) - 2, -1, -1):
            prediction.insert(0, viterbi[t + 1][previous].prev)
            previous = viterbi[t + 1][previous].prev

        return prediction

    def _get_viterbi_matrix(self, probabilities, sentence):
        """
        Creates, fills and returns the matrix according to the Viterbi algorithm
        """
        states = self._lr.classes_
        viterbi = [{}]
        curr_probabilities = probabilities[0]
        for i, current_state in enumerate(states):
            viterbi[0][current_state] = self._ViterbiCell(curr_probabilities[i], None)
        for t in range(1, len(sentence)):
            viterbi.append({})
            for i, current_state in enumerate(states):
                max_tr_prob = -1
                previous_state_selected = None
                for j, previous_state in enumerate(states):
                    curr_probabilities = probabilities[1 + (t - 1) * len(states) + j]
                    curr_state_probability = viterbi[t - 1][previous_state].prob * curr_probabilities[i]
                    if curr_state_probability > max_tr_prob:
                        max_tr_prob = curr_state_probability
                        previous_state_selected = previous_state
                viterbi[t][current_state] = self._ViterbiCell(max_tr_prob, previous_state_selected)
        return viterbi

    _ViterbiCell = namedtuple('_ViterbiCell', ['prob', 'prev'])

    def _get_probabilities(self, sentence):
        """
        :return: An array-like object where each element is an array-like probability array that corresponds to the
        order in self._lr.classes_. The first element contains the probabilities of the first Viterbi column. Then, for
        the t word of the previous state j the corresponding probabilities are at:
            probabilities[1 + (t - 1) * len(states) + j]
        """
        states = self._lr.classes_
        current_annotation = [[word, None] for word in sentence]
        # Since we condition on prior tags, the first annotation is the same for all
        annotated_sentences = [(0, deepcopy(current_annotation))]
        for t in range(1, len(sentence)):
            for previous_state in states:
                current_annotation[t - 1][1] = previous_state
                annotated_sentences.append((t, deepcopy(current_annotation)))
        feature_matrix = self._feature_vectors_from_specified_annotated_sentences(annotated_sentences)
        probabilities = self._lr.predict_proba(feature_matrix)
        return probabilities

    @classmethod
    def _target_vector_from_annotated_sentences(cls, annotated_sentences):
        # noinspection PyProtectedMember
        """
        :return: A target vector where each element is the correct tag for a word

        >>> annotated_sentences = [[('I', 'PRON'), ('love', 'VERB'), ('Greece', 'PROPN')], [('I', 'PRON'),
        ...    ('hate', 'VERB'), ('Greece', 'PROPN')]]
        >>> MemmViterbi._target_vector_from_annotated_sentences(annotated_sentences)
        ['PRON', 'VERB', 'PROPN', 'PRON', 'VERB', 'PROPN']
        """
        return [t for sentence in annotated_sentences for w, t in sentence]

    def _feature_vectors_from_annotated_sentences(self, annotated_sentences):
        """
        :return: A matrix where each row is a feature vector corresponding to a word in the given annotated sentences
        """
        feature_vector_length = self._get_feature_vector_length()

        feature_matrix = lil_matrix((self._number_of_words, feature_vector_length))

        current_global_word_idx = 0
        for annotated_sentence in annotated_sentences:
            safe_annotated_sentence = _SafeTaggedSentence(annotated_sentence)
            for idx in range(len(annotated_sentence)):
                for feature_vector_idx in self._generate_feature_vector_idx(idx, safe_annotated_sentence):
                    feature_matrix[current_global_word_idx, feature_vector_idx] = 1
                current_global_word_idx += 1

        return feature_matrix

    def _feature_vectors_from_specified_annotated_sentences(self, specified_annotated_sentences):
        """
        The same as _feature_vectors_from_annotated_sentences, only that instead of returning the feature vectors for
        all given annotated words, the only feature vectors returned are the specific words that are specified.
        """
        feature_vector_length = self._get_feature_vector_length()

        feature_matrix = lil_matrix((len(specified_annotated_sentences), feature_vector_length))

        for word_idx, (idx, annotated_sentence) in enumerate(specified_annotated_sentences):
            safe_annotated_sentence = _SafeTaggedSentence(annotated_sentence)
            for feature_vector_idx in self._generate_feature_vector_idx(idx, safe_annotated_sentence):
                feature_matrix[word_idx, feature_vector_idx] = 1

        return feature_matrix

    def _get_feature_vector_length(self):
        """
        :return: The length of a feature vector
        """
        num_of_unique_words = len(self._word_to_feature_idx)
        num_of_known_tags = len(self._tag_to_feature_idx)
        number_of_surrounding_words = 5
        number_of_prior_tags = 1
        number_of_suffixes = len(self._suffix_to_feature_idx)
        number_of_prefixes = len(self._prefix_to_feature_idx)
        number_of_word_shape_features = 5
        feature_vector_length = num_of_unique_words * number_of_surrounding_words + \
            num_of_known_tags * number_of_prior_tags + \
            number_of_prefixes + number_of_suffixes + \
            number_of_word_shape_features
        return feature_vector_length

    def _feature_vector(self, idx, annotated_sentence):
        """
        :return: A feature array based on the word itself, those that surround it, and the tags that come before it
        """
        feature_vector = lil_matrix((1, self._get_feature_vector_length()))

        safe_annotated_sentence = _SafeTaggedSentence(annotated_sentence)
        for feature_vector_idx in self._generate_feature_vector_idx(idx, safe_annotated_sentence):
            feature_vector[0, feature_vector_idx] = 1

        return feature_vector

    def _init_word_data(self, annotated_sentences):
        """
        Initializes the word data
        """
        self._word_to_feature_idx = {_BeginningOfSentence: 0, _EndOfSentence: 1}
        self._word_to_count = defaultdict(int)
        self._number_of_words = 0
        self._tag_to_feature_idx = {}
        self._suffix_to_feature_idx = {}
        self._prefix_to_feature_idx = {}
        for sentence in annotated_sentences:
            for i, (word, tag) in enumerate(sentence):
                if i == 0 and tag != 'PROPN':
                    word = word.lower()
                self._number_of_words += 1
                self._word_to_count[word] += 1
                if word not in self._word_to_feature_idx:
                    self._word_to_feature_idx[word] = len(self._word_to_feature_idx)
                if tag not in self._tag_to_feature_idx:
                    self._tag_to_feature_idx[tag] = len(self._tag_to_feature_idx)
                for suffix in self._generate_word_suffixes(word):
                    if suffix in self._suffix_to_feature_idx:
                        break
                    self._suffix_to_feature_idx[suffix] = len(self._suffix_to_feature_idx)
                for prefix in self._generate_word_prefixes(word):
                    if prefix in self._prefix_to_feature_idx:
                        break
                    self._prefix_to_feature_idx[prefix] = len(self._prefix_to_feature_idx)

    @classmethod
    def _generate_word_suffixes(cls, word):
        # noinspection PyProtectedMember
        """
        >>> list(MemmViterbi._generate_word_suffixes('abcd'))
        ['abcd', 'bcd', 'cd', 'd']
        """
        word_length = len(word)
        start_idx = max(0, word_length - cls._suffix_max_length)
        for idx in range(start_idx, word_length):
            yield word[idx:]

    @classmethod
    def _generate_word_prefixes(cls, word):
        # noinspection PyProtectedMember
        """
        >>> list(MemmViterbi._generate_word_prefixes('abcd'))
        ['abcd', 'abc', 'ab', 'a']
        """
        word_length = len(word)
        stop_idx = min(cls._prefix_max_length, word_length)
        for idx in range(stop_idx, 0, -1):
            yield word[:idx]

    def _generate_feature_vector_idx(self, idx, safe_annotated_sentence):
        # noinspection PyProtectedMember
        """
        Generates the non-zero indexes of the feature vector word at safe_annotated_sentence[idx][0]
        """
        current_offset = 0
        num_of_known_words = len(self._word_to_feature_idx)
        num_of_known_tags = len(self._tag_to_feature_idx)
        word = safe_annotated_sentence[idx][0]

        try:
            yield self._word_to_feature_idx[word]
        except KeyError:
            pass
        current_offset += num_of_known_words

        try:
            yield current_offset + self._word_to_feature_idx[safe_annotated_sentence[idx - 1][0]]
        except KeyError:
            pass
        current_offset += num_of_known_words

        try:
            yield current_offset + self._word_to_feature_idx[safe_annotated_sentence[idx - 2][0]]
        except KeyError:
            pass
        current_offset += num_of_known_words

        try:
            yield current_offset + self._word_to_feature_idx[safe_annotated_sentence[idx + 1][0]]
        except KeyError:
            pass
        current_offset += num_of_known_words

        try:
            yield current_offset + self._word_to_feature_idx[safe_annotated_sentence[idx + 2][0]]
        except KeyError:
            pass
        current_offset += num_of_known_words

        if safe_annotated_sentence[idx - 1][1] is not None:
            yield current_offset + self._tag_to_feature_idx[safe_annotated_sentence[idx - 1][1]]
        current_offset += num_of_known_tags

        if not self._is_rare_word(word):
            return

        for prefix in self._generate_word_prefixes(word):
            try:
                yield current_offset + self._prefix_to_feature_idx[prefix]
            except KeyError:
                pass
        current_offset += len(self._prefix_to_feature_idx)

        for suffix in self._generate_word_suffixes(word):
            try:
                yield current_offset + self._suffix_to_feature_idx[suffix]
            except KeyError:
                pass
        current_offset += len(self._suffix_to_feature_idx)

        if any(ch.isdigit() for ch in word):
            yield current_offset
        current_offset += 1

        if any(ch.isupper() for ch in word):
            yield current_offset
        current_offset += 1

        if word.isupper():
            yield current_offset
        current_offset += 1

        if '-' in word:
            yield current_offset
        current_offset += 1

        if self._upper_case_with_digit_and_dash_re.match(word) is not None:
            yield current_offset

    def _is_rare_word(self, word):
        return self._word_to_count[word] < self._not_unknown_count_threshold


class _BeginningOfSentence:
    pass


class _EndOfSentence:
    pass


class _SafeTaggedSentence:
    """
    >>> sl = _SafeTaggedSentence([1, 2, 3])
    >>> sl[0]
    1
    >>> sl[-1]
    (<class 'solution12A._BeginningOfSentence'>, None)
    >>> sl[3]
    (<class 'solution12A._EndOfSentence'>, None)

    """

    def __init__(self, l):
        self._l = l

    def __getitem__(self, idx):
        if idx < 0:
            return _BeginningOfSentence, None
        elif idx >= len(self._l):
            return _EndOfSentence, None

        return self._l[idx]

from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []
        for sentence in parallel_corpus:
            posteriors.append(self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)] / np.sum(self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)], axis=0))
        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            eblo ahahaha: the value of evidence lower bound
            чё, катаешь? ладно, катай, только не спались
        """
        eblo = 0
        for i in range(len(parallel_corpus)):
            eblo += (posteriors[i] * np.log(self.translation_probs[np.ix_(parallel_corpus[i].source_tokens, parallel_corpus[i].target_tokens)] / (len(parallel_corpus[i].source_tokens) * posteriors[i]))).sum()
        return eblo
    

    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs *= 0 
        for i in range(len(parallel_corpus)):
            np.add.at(self.translation_probs, (parallel_corpus[i].source_tokens[:, None], parallel_corpus[i].target_tokens[None, :]), posteriors[i])
        self.translation_probs /= np.sum(self.translation_probs, axis=1)[:, None]
        return self._compute_elbo(parallel_corpus, posteriors)


    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        predict = []
        for sentence in sentences:
            source_tokens = np.argmax(self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)], axis=0) + 1
            predict.append(list(zip(source_tokens, np.arange(1, len(sentence.target_tokens) + 1, dtype=np.int32))))
        return predict


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        try:
            return self.alignment_probs[(src_length, tgt_length)]
        except:
            self.alignment_probs[(src_length, tgt_length)] = np.ones((src_length, tgt_length), dtype=np.float32) / src_length
            return self.alignment_probs[(src_length, tgt_length)]
        # if (src_length, tgt_length) not in self.alignment_probs:
        #     self.alignment_probs[(src_length, tgt_length)] = np.ones((src_length, tgt_length), dtype=np.float32) / src_length
        # return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
        posteriors = []
        for sentence in parallel_corpus:
            src = sentence.source_tokens
            trg = sentence.target_tokens
            tmp1 = self.translation_probs[np.ix_(src, trg)]
            tmp2 = self._get_probs_for_lengths(len(src), len(trg))
            tmp = tmp1 * tmp2
            posteriors.append((tmp) / np.sum(tmp, axis=0))
        return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
        eblo = 0
        for i in range(len(posteriors)):
            src = parallel_corpus[i].source_tokens
            trg = parallel_corpus[i].target_tokens
            eblo += (posteriors[i] *\
                 (self.log(self.translation_probs[np.ix_(src, trg)]) - self.log(posteriors[i]) + self.log(self._get_probs_for_lengths(len(src), len(trg))))).sum()
        return eblo

    def _m_step(self, parallel_corpus, posteriors):
        num = {}
        self.translation_probs *= 0 
        for i in range(len(parallel_corpus)):
            src = parallel_corpus[i].source_tokens
            trg = parallel_corpus[i].target_tokens
            np.add.at(self.translation_probs, (src[:, None], trg[None, :]), posteriors[i])
            if (len(src), len(trg)) not in num:
                num[(len(src), len(trg))] = 1
                self.alignment_probs[(len(src), len(trg))] = posteriors[i]
            else:
                self.alignment_probs[(len(src), len(trg))] =  (self.alignment_probs[(len(src), len(trg))] * num[(len(src), len(trg))] + posteriors[i]) / (num[(len(src), len(trg))] + 1)
                num[(len(src), len(trg))] += 1
        self.translation_probs /= np.sum(self.translation_probs, axis=1)[:, None]
        return self._compute_elbo(parallel_corpus, posteriors)
    @staticmethod
    def log(M):
        return np.log(np.where(M == 0, 1, M))

from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    tree = open(filename).read().replace('&', '&amp;')
    root = ET.fromstring(tree)
    sentence = []
    la = []
    for child in root:
        sentence.append(SentencePair(child.find('english').text.split(), child.find('czech').text.split()))
        s = child.find('sure').text
        sure = []
        if s is not None:
            sure = [tuple(map(int, el.split('-'))) for el in s.split()]
        possible = []
        p = child.find('possible').text
        if p is not None:
            possible = [tuple(map(int, el.split('-'))) for el in p.split()]
        la.append(LabeledAlignment(sure, possible))
    return sentence, la


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    t_idx_src = {}
    t_idx_tgt = {}

    if freq_cutoff is not None:
        src_freq = {}
        trg_freq = {}

    for s in sentence_pairs:
        for se in s.source:
            if se not in t_idx_src:
                t_idx_src[se] = len(t_idx_src)
                if freq_cutoff is not None:
                    src_freq[se] = 1
            else:
                if freq_cutoff is not None:
                    src_freq[se] += 1
        for sc in s.target:
            if sc not in t_idx_tgt:
                t_idx_tgt[sc] = len(t_idx_tgt)
                if freq_cutoff is not None:
                    trg_freq[sc] = 1
            else:
                if freq_cutoff is not None:
                    trg_freq[sc] += 1

    if freq_cutoff is None:
        return t_idx_src, t_idx_tgt
    src_freq = dict(sorted(src_freq.items(), key=lambda i: i[1], reverse=True)[:freq_cutoff])
    trg_freq = dict(sorted(trg_freq.items(), key=lambda i: i[1], reverse=True)[:freq_cutoff])
    for key in src_freq.keys():
        src_freq[key] = t_idx_src[key]
    for key in trg_freq.keys():
        trg_freq[key] = t_idx_tgt[key]
    return src_freq, trg_freq


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for s in sentence_pairs:
        english = []
        s_flag = True
        for se in s.source:
            if se in source_dict:
                english.append(source_dict[se])
            else:
                s_flag = False
                break
        czech = []
        c_flag = True
        for sc in s.target:
            if sc in target_dict:
                czech.append(target_dict[sc])
            else:
                c_flag = False
                break
        if s_flag and c_flag:
            tokenized_sentence_pairs.append(TokenizedSentencePair(np.array(english), np.array(czech)))
    return tokenized_sentence_pairs

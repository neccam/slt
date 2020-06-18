# coding: utf-8
import numpy as np

from collections import defaultdict, Counter
from typing import List
from torchtext.data import Dataset

SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self):
        # don't rename stoi and itos since needed for torchtext
        # warning: stoi grows with unknown tokens, don't use for saving or size
        self.specials = []
        self.itos = []
        self.stoi = None
        self.DEFAULT_UNK_ID = None

    def _from_list(self, tokens: List[str] = None):
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self.stoi) == len(self.itos)

    def _from_file(self, file: str):
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r", encoding="utf-8") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str):
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        with open(file, "w", encoding="utf-8") as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]):
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self.itos)
            # add to vocab if not already there
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.stoi[token] == self.DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.itos)


class TextVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None):
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 0
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def array_to_sentence(self, array: np.array, cut_at_eos=True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.array, cut_at_eos=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(self.array_to_sentence(array=array, cut_at_eos=cut_at_eos))
        return sentences


class GlossVocabulary(Vocabulary):
    def __init__(self, tokens: List[str] = None, file: str = None):
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        super().__init__()
        self.specials = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        self.DEFAULT_UNK_ID = lambda: 1
        self.stoi = defaultdict(self.DEFAULT_UNK_ID)

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

        # TODO (Cihan): This bit is hardcoded so that the silence token
        #   is the first label to be able to do CTC calculations (decoding etc.)
        #   Might fix in the future.
        assert self.stoi[SIL_TOKEN] == 0

    def arrays_to_sentences(self, arrays: np.array) -> List[List[str]]:
        gloss_sequences = []
        for array in arrays:
            sequence = []
            for i in array:
                sequence.append(self.itos[i])
            gloss_sequences.append(sequence)
        return gloss_sequences


def filter_min(counter: Counter, minimum_freq: int):
    """ Filter counter by min frequency """
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    """ Cut counter to most frequent,
    sorted numerically and alphabetically"""
    # sort by frequency, then alphabetically
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens


def build_vocab(
    field: str, max_size: int, min_freq: int, dataset: Dataset, vocab_file: str = None
) -> Vocabulary:
    """
    Builds vocabulary for a torchtext `field` from given`dataset` or
    `vocab_file`.

    :param field: attribute e.g. "src"
    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param dataset: dataset to load data for field from
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :return: Vocabulary created from either `dataset` or `vocab_file`
    """

    if vocab_file is not None:
        # load it from file
        if field == "gls":
            vocab = GlossVocabulary(file=vocab_file)
        elif field == "txt":
            vocab = TextVocabulary(file=vocab_file)
        else:
            raise ValueError("Unknown vocabulary type")
    else:
        tokens = []
        for i in dataset.examples:
            if field == "gls":
                tokens.extend(i.gls)
            elif field == "txt":
                tokens.extend(i.txt)
            else:
                raise ValueError("Unknown field type")

        counter = Counter(tokens)
        if min_freq > -1:
            counter = filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        if field == "gls":
            vocab = GlossVocabulary(tokens=vocab_tokens)
        elif field == "txt":
            vocab = TextVocabulary(tokens=vocab_tokens)
        else:
            raise ValueError("Unknown vocabulary type")

        assert len(vocab) <= max_size + len(vocab.specials)
        assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN

    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab

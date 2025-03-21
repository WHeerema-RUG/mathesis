# Algorithm 1 (adapted)
# from Neural Machine Translation of Rare Words with Subword Units
# Authors: Rico Sennrich, Barry Haddow and Alexandra Birch
# School of Informatics, University of Edinburgh
# Last paper revision 10/06/2016

from collections import Counter, defaultdict
from itertools import chain
from nltk.tokenize import MWETokenizer 
import re


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def bpe_alg1(sents, merges):
    """Performs merges and tokenizes the text"""
    # Convert vocabulary to mergeable format (mine)
    # EOW = #
    vocab_counter = Counter([token.lower()+"#" for token
                             in re.findall("\w+", " ".join(sents))])
    vocab = {" ".join(list(key)): value
             for key, value in vocab_counter.items()}
    # Do merges (from paper)
    for _ in range(merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    # Tokenize text (mine)
    # Treats BPs as multi-word expressions
    vocab_ordered = sorted(set(chain.from_iterable([merged.split() for merged
                                                    in vocab.keys()])),
                           key=len, reverse=True)
    print(vocab_ordered)
    bpe_tokenizer = MWETokenizer([tuple(word) for word in vocab_ordered],
                                 separator="")
    # Replace non-words with EOW markers, tokenize
    tok_sents = [bpe_tokenizer.tokenize(list(re.sub("\W+", "#", line)))
                 for line in sents]
    # Turn into BPE IDs
    vocab_indices = {bp: vocab_ordered.index(bp) for bp in vocab_ordered}
    id_sents = [[vocab_indices[sub] for sub in sent] for sent in tok_sents]
    # Return tokenized sentences and vocabulary
    return id_sents, vocab_indices

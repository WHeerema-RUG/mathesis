# BPE/TRANSFORMER EVALUATOR
# For the Master's thesis project
# Performs BPE merges and transformer training for learning ease metrics.
# Author: Wessel Heerema
# Date: 25/02/2025

import argparse
# Credit to soaxelbrooke et al. on GitHub
from bpe import Encoder
from collections import Counter
from itertools import chain
from scipy.stats import entropy

from transformer_eval import transformer_ops


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Sentence input")
    parser.add_argument("-m", "--merges", type=int, default=400,
                        help="Amount of BPE merges to perform"
                        "(default 400)")
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="Amount of epochs to train the transformer for"
                        "(default 20)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Output every epoch for the transformer")
    args = parser.parse_args()
    return args


def bpe_ifier(sents, merges):
    """Do BPE operations for a set of sentences"""
    # Do BPE merges and tokenization
    bpe = Encoder(merges, pct_bpe=0.9)
    bpe.fit(sents)
    # Get vocab size
    vocab_size = bpe.vocabs_to_dict()["kwargs"]["vocab_size"]
    # Get tokenized sentences for Shannon entropy measure
    returnable = [next(bpe.transform([sent])) for sent in sents]
    # Get frequency for all types in every sentence
    dim1_tokens = Counter(list(chain.from_iterable(returnable))).values()
    # Get probabilities for subword occurrence
    prob_array = [value/sum(dim1_tokens) for value in dim1_tokens]
    # Do entropy measure
    print("Shannon entropy:", entropy(prob_array))
    # Return tokenized sentences
    return returnable, vocab_size


def main(args):
    # Load file
    with open(args.infile, "r") as f1:
        sents_raw = [line[:-1] for line in f1.readlines()]
    # Do BPE
    sents_tok, vocab = bpe_ifier(sents_raw, args.merges)
    # Do transformer from another file
    transformer_ops(sents_tok, vocab, args.epochs)


if __name__ == "__main__":
    main(create_args())

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
    parser.add_argument("-s", "--sennrich", action="store_true",
                        help="Use Algorithm 1 by Sennrich et al.")
    args = parser.parse_args()
    return args


def evaluate_bpe(tok_list, vocab_bps, vocab_outfile="none"):
    """Calculate Shannon entropy and, if specified, write out vocab"""
    # Get frequency for all types in every sentence
    dim1_tokens = Counter(list(chain.from_iterable(tok_list))).values()
    # Get probabilities for subword occurrence
    prob_array = [value/sum(dim1_tokens) for value in dim1_tokens]
    # Do entropy measure
    entropy_no = entropy(prob_array)
    print("Shannon entropy:", entropy_no)
    # Save vocabulary to output file if specified
    if vocab_outfile != "none":
        with open(vocab_outfile, "w") as f1:
            f1.write("\n".join(vocab_bps))
    return entropy_no


def bpe_ifier(sents, merges):
    """Do BPE operations for a set of sentences"""
    # Do BPE merges and tokenization
    bpe = Encoder(merges, pct_bpe=0.9)
    bpe.fit(sents)
    # Get vocabulary
    vocab_dict = bpe.vocabs_to_dict()
    # Get tokenized sentences for Shannon entropy measure
    returnable = [next(bpe.transform([sent])) for sent in sents]
    # Return tokenized sentences and vocabulary
    return returnable, vocab_dict


def main(args):
    # Load file
    with open(args.infile, "r") as f1:
        sents_raw = [line[:-1] for line in f1.readlines()]
    # Do BPE
    if args.sennrich:
        from sennrich_etal import bpe_alg1
        sents_tok, vocab = bpe_alg1(sents_raw, args.merges)
        print(sents_tok)
        # Calculate entropy
        evaluate_bpe(sents_tok, vocab.keys())
        transformer_ops(sents_tok, len(vocab), args.epochs)
    else:
        sents_tok, vocab = bpe_ifier(sents_raw, args.merges)
        # Calculate entropy
        evaluate_bpe(sents_tok, vocab["byte_pairs"])
        # Do transformer from another file
        transformer_ops(sents_tok, vocab["kwargs"]["vocab_size"], args.epochs)


if __name__ == "__main__":
    main(create_args())

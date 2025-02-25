# BPE/TRANSFORMER EVALUATOR
# For the Master's thesis project
# Performs BPE merges and transformer training for learning ease metrics.
# Author: Wessel Heerema
# Date: 25/02/2025

import argparse
# Credit to soaxelbrooke et al. on GitHub
from bpe import Encoder
from scipy.stats import entropy


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Sentence input")
    parser.add_argument("-m", "--merges", type=int, default=400,
                        help="Amount of BPE merges to perform"
                        "(default 400)")
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="Amount of epochs to train the transformer for"
                        "(default 20)")
    args = parser.parse_args()
    return args


def bpe_ifier(sents, merges):
    """Do BPE operations for a set of sentences"""
    # Do BPE merges and tokenization
    bpe = Encoder(merges, pct_bpe=1)
    bpe.fit(sents)
    # Pull vocab for Shannon entropy measure
    vocab = bpe.vocabs_to_dict()
    # Get probabilities for each subword occurring
    prob_array = [value/vocab["kwargs"]["vocab_size"]
                  for _, value in vocab["words"]]
    # Do entropy measure
    print("Shannon entropy:", entropy(prob_array))
    # Return tokenized sentences
    return [bpe.tokenize(sent) for sent in sents]


def main(args):
    # Load file
    with open(args.infile, "r") as f1:
        sents_raw = [line[:-1] for line in f1.readlines()]
    # Do BPE
    sents_tok = bpe_ifier(sents_raw, args.merges)
    # TODO: Do transformer


if __name__ == "__main__":
    main(create_args())

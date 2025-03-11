# MAIN PROGRAM
# For the Master's thesis project
# Does a batch generation and evaluation of every language.
# Assumes proto-sentences, lexicon and feature occurrences are already
# generated.
# Author: Wessel Heerema
# Date: 11/03/2025

# Import "proper" modules for I/O
import argparse
import json
import os
import sys

# Import custom modules
from sentences import renderer as ren
import evaluate as eval


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="Get lexicon from file")
    parser.add_argument("feats", type=str,
                        help="Import feature probabilities from file")
    parser.add_argument("grammar", type=str,
                        default="sentences/agglutinative.json",
                        help="Load grammar for the sentences; serves as a "
                        "template if orthography is specified")
    parser.add_argument("-o", "--orthography", type=str,
                        default="empty",
                        help="Specify a standardized orthography for all "
                        "grammars, which enables grammatical variation"
                        "(default empty)")
    parser.add_argument("-s", "--sentcap", type=int, default=100,
                        help="Cap on the amount of sentences to use"
                        "(default 100)")
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


def batch_eval(sents, grammar, particles, orthography,
               merges, epochs, path, export_append,
               verbose=False):
    """Do a batch evaluation on the whole set"""
    # Render every sentence
    parsed = [ren.render_sent(sent, grammar, particles, orthography)
              for sent in sents]
    # Output to text file for control
    with open(path + "parse-" + export_append + ".txt", "w") as fo:
        fo.write("\n".join(parsed))
    # Calculate BPE
    sents_tok, vocab = eval.bpe_ifier(parsed, merges)
    # Train transformer
    eval.transformer_ops(sents_tok, vocab, epochs, verbose=verbose)


def main(args):
    # Import files
    sents, grammar, particles = ren.load_data(args.infile, args.grammar,
                                              args.lexicon)
    # Get the program path
    # Credit to neuro and Asclepius on StackOverflow
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    try:
        os.mkdir(path+"/data")
    except FileExistsError:
        pass
    if args.orthography == "empty":
        # If no orthography specified, do an orthography run
        orths = ["none"]
        # Fetch all orthographies
        # Credit to Nadia Alramli on StackOverflow
        for root, _, files in os.walk(path+"/sentences"):
            for name in files:
                if "orth" in name:
                    # If an orthography, add to the iterable
                    orths.append(os.path.join(root, name))
        # Iterate through all available orthographies
        for orth_path in orths:
            # Load orthography
            if orth_path == "none":
                orthography = {}
            else:
                with open(orth_path, "r") as fs:
                    orthography = json.load(fs)
            # Carry out evaluation with this setup
            batch_eval(sents, grammar, particles, orthography,
                       path=path+"/data/", export_append=orth_path[:-5],
                       verbose=args.verbose)
    else:
        # If orthography specified, do a morphology run
        with open(args.orthography, "r") as fs:
            orthography = json.load(fs)
        # The grammar can be expressed as a 7-bit number
        # (when including fusional)
        # Iterate through the first 6 bits to capture all (agglutinative)
        # grammars, starting with analytical
        for i in range(2**6):
            # Convert the number to Boolean grammar values
            binary = [True if d == "1" else False for d in format(i, "06b")]
            # Assign every value
            grammar["marking"]["definite"] = binary[0]
            grammar["marking"]["indefinite"] = binary[1]
            grammar["marking"]["past"] = binary[2]
            grammar["marking"]["non-past"] = binary[3]
            grammar["marking"]["singular"] = binary[4]
            grammar["marking"]["plural"] = binary[5]
            # Carry out evaluation with this setup
            batch_eval(sents, grammar, particles, orthography,
                       path=path+"/data/", export_append=str(i),
                       verbose=args.verbose)
        # Next, invert fusional option, with every marking enabled
        grammar["fusional"] = not grammar["fusional"]
        # Final evaluation with the fusional grammar
        batch_eval(sents, grammar, particles, orthography,
                   path=path+"/data/", export_append="fusional",
                   verbose=args.verbose)


if __name__ == "__main__":
    main(create_args())

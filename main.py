# MAIN PROGRAM
# For the Master's thesis project
# Does a batch generation and evaluation of every language.
# Assumes proto-sentences and lexicon are already generated.
# Author: Wessel Heerema
# Date: 11/03/2025

# Import "proper" modules for I/O
import argparse
import json
import pandas as pd
import os
import sys

# Import custom modules
from sentences import renderer as ren
import evaluate as eval


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="Get gloss sentences from file")
    parser.add_argument("grammar", type=str,
                        default="sentences/agglutinative.json",
                        help="Load grammar for the sentences; serves as a "
                        "template if orthography is specified")
    parser.add_argument("lexicon", type=str,
                        help="Get lexicon from file")
    parser.add_argument("-s", "--orthography", type=str,
                        default="empty",
                        help="Specify a standardized orthography for all "
                        "grammars, which enables grammatical variation"
                        "(default empty)")
    parser.add_argument("-n", "--sentcap", type=int, default=9999999999,
                        help="Cap on the amount of sentences to use"
                        "(default 9999999999)")
    parser.add_argument("-m", "--merges", type=int, default=400,
                        help="Amount of BPE merges to perform"
                        "(default 400)")
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="Amount of epochs to train the transformer for"
                        "(default 20)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Output every epoch for the transformer")
    parser.add_argument("-o", "--outfile", type=str,
                        default="results.csv",
                        help=".csv file to export all results to"
                        "(default results.csv)")
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
    # Calculate BPE and save vocabulary
    bpe_out = path + "bpe-" + export_append + ".txt"
    sents_tok, vocab, entropy = eval.bpe_ifier(parsed, merges,
                                               bpe_out)
    # Train transformer
    loss, perplexity = eval.transformer_ops(sents_tok, vocab, epochs,
                                            verbose=verbose)
    return {"ID": export_append, "Entropy": entropy, "Loss": loss,
            "Perplexity": perplexity}


def prepare_export(parent, child):
    """Process child dict for parent dict export"""
    for key, value in child.items():
        parent[key].append(value)


def main(args):
    # Import files
    sents, grammar, particles = ren.load_data(args.infile, args.grammar,
                                              args.lexicon)
    # Set sentence cap
    if len(sents) > args.sentcap:
        sents = sents[:args.sentcap]
    # Get the program path
    # Credit to neuro and Asclepius on StackOverflow
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    try:
        os.mkdir(path+"/data")
    except FileExistsError:
        pass
    df_temp = {"ID": [], "Entropy": [], "Loss": [], "Perplexity": []}
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
                export_append = "none"
            else:
                with open(orth_path, "r") as fs:
                    orthography = json.load(fs)
                export_append = os.path.basename(orth_path)[:-5]
            # Carry out evaluation with this setup
            print("="*25, export_append.upper(), "="*25)
            out = batch_eval(sents, grammar, particles, orthography,
                             merges=args.merges, epochs=args.epochs,
                             path=path+"/data/", export_append=export_append,
                             verbose=args.verbose)
            # Append to other results
            prepare_export(df_temp, out)
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
            print("="*25, format(i, "06b"), "="*25)
            out = batch_eval(sents, grammar, particles, orthography,
                             merges=args.merges, epochs=args.epochs,
                             path=path+"/data/", export_append=str(i),
                             verbose=args.verbose)
            # Append to other results
            prepare_export(df_temp, out)
        # Next, invert fusional option, with every marking enabled
        grammar["fusional"] = not grammar["fusional"]
        # Final evaluation with the fusional grammar
        print("="*25, "FUSIONAL - ALL ENABLED", "="*25)
        out = batch_eval(sents, grammar, particles, orthography,
                         merges=args.merges, epochs=args.epochs,
                         path=path+"/data/", export_append="fusional",
                         verbose=args.verbose)
        # Append to other results
        prepare_export(df_temp, out)
    # Export results to CSV
    df = pd.DataFrame(df_temp)
    df.to_csv(args.outfile, index=False)
    print("Exported all results to", args.outfile)


if __name__ == "__main__":
    main(create_args())

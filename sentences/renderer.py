# SENTENCE RENDERER
# For the Master's thesis project
# Renders proto-sentences in a particular grammar.
# Author: Wessel Heerema
# Date: 20/02/2025

import argparse
import json


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="Get gloss sentences from JSON file")
    parser.add_argument("grammar", type=str,
                        help="Get grammar from JSON file")
    parser.add_argument("lexicon", type=str,
                        help="Get lexicon from JSON file")
    parser.add_argument("-o", "--outfile", type=str,
                        default="none",
                        help="Export sentences to file"
                        "(default [grammar].txt)")
    args = parser.parse_args()
    return args


def render_sent(protosent, grammar, particles):
    """Render a single sentence according to a grammar"""
    # Initialize output sentence
    outsent = []
    # Iterate through all words
    for gloss in protosent:
        # Look for dropout
        marked = []
        for particle in gloss[1:]:
            if grammar["marking"][particle]:
                marked.append(particle)
        # Handle grammar
        if grammar["fusional"]:
            # If fusional, get particle that corresponds to both
            # TODO: Add fusional grammar to lexicon
            raise NotImplementedError("Fusional grammar not yet supported")
            word = gloss[0] + particles["&".join(marked)]
        else:
            # If agglutinative, join everything together
            word = gloss[0] + "".join([particles[valid]
                                       for valid in marked])
        # Add to sentence
        outsent.append(word)
    # Return capitalized sentence with a period at the end
    return " ".join(outsent).capitalize() + "."


def main(args):
    # Import files
    with open(args.infile, "r") as fs:
        sents = json.load(fs)
    print("Imported sentences from", args.infile)
    with open(args.grammar, "r") as fg:
        grammar = json.load(fg)
    print("Imported grammar from", args.grammar)
    # Process lexicon to only get the particles
    with open(args.lexicon, "r") as fl:
        particles = {value: key for key, value
                     in json.load(fl)["grammar"].items()}
    print("Imported particles from", args.lexicon)
    # Iterate through all sentences
    parsed = [render_sent(sent, grammar, particles)
              for sent in sents]
    # Output to text file
    if args.outfile == "none":
        outpath = args.grammar[:args.grammar.index(".")] + ".txt"
    else:
        outpath = args.outfile
    with open(outpath, "w") as fo:
        fo.write("\n".join(parsed))
    print("Exported sentences to", outpath)


if __name__ == "__main__":
    main(create_args())

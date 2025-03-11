# SENTENCE RENDERER
# For the Master's thesis project
# Renders proto-sentences in a particular grammar.
# Author: Wessel Heerema
# Date: 20/02/2025

import argparse
import json
import re


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="Get gloss sentences from JSON file")
    parser.add_argument("grammar", type=str,
                        help="Get grammar from JSON file")
    parser.add_argument("lexicon", type=str,
                        help="Get lexicon from JSON file")
    parser.add_argument("-s", "--orthography", type=str,
                        default="none",
                        help="Orthography for respelling"
                        "(default none)")
    parser.add_argument("-o", "--outfile", type=str,
                        default="none",
                        help="Export sentences to file"
                        "(default [grammar].txt)")
    args = parser.parse_args()
    return args


def load_data(sents, grammar, lexicon):
    """Load all the static data for sentence rendering"""
    # Import files
    with open(sents, "r") as fs:
        sents_json = json.load(fs)
    print("Imported sentences from", sents)
    with open(grammar, "r") as fg:
        grammar_json = json.load(fg)
    print("Imported grammar from", grammar)
    # Process lexicon to only get the particles
    with open(lexicon, "r") as fl:
        particles_json = {value: key for key, value
                     in json.load(fl)["grammar"].items()}
    print("Imported particles from", lexicon)
    return sents_json, grammar_json, particles_json


def ortho_transform(word, orthography):
    """Respell a word in a given orthography"""
    for key, value in orthography.items():
        word = re.sub(key, value, word)
    return word


def render_sent(protosent, grammar, particles, orthography):
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
            # Expects morpheme in the format of feat&number
            word = gloss[0] + particles["&".join(marked)]
        else:
            # If agglutinative, join everything together
            word = gloss[0] + "".join([particles[valid]
                                       for valid in marked])
        # Respell and add to sentence
        outsent.append(ortho_transform(word, orthography))
    # Return capitalized sentence with a period at the end
    return " ".join(outsent).capitalize() + "."


def main(args):
    # Load all files
    sents, grammar, particles = load_data(args.infile, args.grammar,
                                          args.lexicon)
    if args.orthography != "none":
        # If orthography specified, load it
        with open(args.orthography, "r") as fs:
            orthography = json.load(fs)
        print("Imported orthography from", args.orthography)
    else:
        # If not, simply treat orthography as empty
        orthography = {}
    # Iterate through all sentences
    parsed = [render_sent(sent, grammar, particles, orthography)
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

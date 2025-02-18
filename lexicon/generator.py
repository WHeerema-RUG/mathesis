# LEXICON GENERATOR
# For the Master's thesis project
# Generates a lexicon on the basis of the phonology file.
# Only needs to be run once, as to control vocabulary.
# Author: Wessel Heerema
# Date: 18/02/2025

import argparse
from collections import defaultdict
import json
from nltk.corpus import brown


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=str,
                        default="phonology.json",
                        help="Get phonology from file"
                        "(default phonology.json)")
    parser.add_argument("-o", "--outfile", type=str,
                        default="lexicon.json",
                        help="Export lexicon to file"
                        "(default lexicon.json)")
    parser.add_argument("-e", "--featexport", type=str, default="none",
                        help="Export feature probabilities to file"
                        "(default none)")
    parser.add_argument("-f", "--featimport", type=str, default="none",
                        help="Import feature probabilities from file"
                        "(default none)")
    args = parser.parse_args()
    return args


def feature_tally(tagged):
    """Look through tagged words and return two dicts of word lengths and
    feature probabilities. Requires Brown-style tags to work
    """
    # Initialize dicts for collection
    word_lens = {"noun": defaultdict(int), "verb": defaultdict(int)}
    feat_occs = {"definite": 0, "indefinite": 0,
                 "past": 0, "non-past": 0,
                 "singular": 0, "plural": 0}
    # Tag and word sets to look for in feature occurrences
    def_tags = {"AT", "CD", "DT", "DTI", "DTS", "DTX"}
    indef_words = {"a", "an", "one", "some", "any"}
    past_tags = {"VBD", "VBN"}
    plural_tags = {"NNS", "NNS$", "NPS", "NPS$", "NRS"}
    # Iterate through the corpus
    for word in tagged:
        # Get rid of hyphenated POS tags for ease of processing
        try:
            pos = word[1][:word[1].index("-")]
        except ValueError:
            pos = word[1]
        # Skip tags with no relevance to tally
        if len(pos) == 0:
            continue
        elif pos not in (def_tags or past_tags or plural_tags) \
            and pos[0] != "N" and pos[0] != "V":
            continue

        # Accumulate word lengths for nouns and verbs
        # To check, use the first character of the POS tag
        if pos[0] == "N":
            word_lens["noun"][len(word[0])] += 1
        elif pos[0] == "V":
            word_lens["verb"][len(word[0])] += 1

        # Check for features
        # Definiteness
        if pos in def_tags:
            if word[0].lower() in indef_words:
                feat_occs["indefinite"] += 1
            else:
                feat_occs["definite"] += 1
        # Tense
        if pos in past_tags:
            feat_occs["past"] += 1
        elif pos[0] == "V":
            feat_occs["non-past"] += 1
        # Plurality
        if pos in plural_tags:
            feat_occs["plural"] += 1
        elif pos[0] == "N":
            feat_occs["singular"] += 1
    # Once all done, return everything
    return word_lens, feat_occs


def main(args):
    # Load or generate feature occurrences
    if args.featimport != "none":
        with open(args.featimport, "r") as ff:
            feat_occs = json.load(ff)
        word_lens = {"noun": feat_occs.pop("noun"),
                     "verb": feat_occs.pop("verb")}
        print("Imported from", args.featimport)
    else:
        word_lens, feat_occs = feature_tally(brown.tagged_words())
        if args.featexport != "none":
            # Thanks to phihag and Mateen Ulhaq on StackOverflow
            with open(args.featexport, "w", encoding="utf-8") as ff:
                json.dump(dict(feat_occs, **word_lens), ff,
                          ensure_ascii=False, indent=4)
            print("Exported to", args.featexport)


if __name__ == "__main__":
    main(create_args())

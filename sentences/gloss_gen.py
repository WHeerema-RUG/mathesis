# GLOSS SENTENCE GENERATOR
# For the Master's thesis project
# Generates proto-sentences with glosses for adaptability.
# Author: Wessel Heerema
# Date: 19/02/2025

import argparse
import json
import random


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lexicon", type=str, help="Lexicon input")
    parser.add_argument("feats", type=str,
                        help="Feature occurrence input")
    parser.add_argument("-o", "--outfile", type=str,
                        default="gloss_sents.json",
                        help="Export sentences to file"
                        "(default gloss_sents.json)")
    parser.add_argument("-n", "--sentcount", type=int, default=100,
                        help="Amount of sentences to generate"
                        "(default 100)")
    args = parser.parse_args()
    return args


def determine_morpho(phrase, options, thres):
    """Determine the morphology given the options"""
    for key, value in options.items():
        if random.random() < thres[key]:
            phrase.append(value[0])
        else:
            phrase.append(value[1])
    return phrase


def generate_gloss(lexicon, options, thres):
    """Generate a proto-sentence with gloss tuples"""
    # Get predicate verb and morphology
    predicate = [random.choice(list(lexicon["verb"].keys()))]
    predicate = determine_morpho(predicate, options["verbs"], thres)
    # Determine sentence transitivity
    transitivity = lexicon["verb"][predicate[0]]
    # If transitive, allow for fewer than the max arguments
    if transitivity > 1:
        transitivity = random.randint(1, transitivity)
    # Add morphology to all nouns
    for i in range(transitivity + 1):
        noun = [random.choice(lexicon["noun"])]
        noun = determine_morpho(noun, options["nouns"], thres)
        # Make subject congruent with predicate
        if i == 0:
            noun[-1] = predicate[-1]
            # Create sentence
            sentence = [tuple(noun), tuple(predicate)]
        else:
            # Add arguments to sentence
            sentence.append(tuple(noun))
    return sentence


def main(args):
    # Import all necessary files
    with open(args.lexicon, "r") as fl:
        lexicon = json.load(fl)
    print("Imported lexicon from", args.lexicon)
    with open(args.feats, "r") as ff:
        feat_occs = json.load(ff)
    print("Imported features from", args.feats)
    # Set options and thresholds
    options = {"nouns": {"definite": ["definite", "indefinite"],
                         "plural": ["plural", "singular"]},
               "verbs": {"past": ["past", "non-past"],
                         "plural": ["plural", "singular"]}}
    thres = {}
    for key, value in dict(options["nouns"], **options["verbs"]).items():
        thres[key] = feat_occs[value[0]] / sum([feat_occs[option]
                                                for option in value])
    # Create gloss sentences
    proto_sents = [generate_gloss(lexicon, options, thres)
                   for _ in range(args.sentcount)]
    with open(args.outfile, "w", encoding="utf-8") as fg:
        json.dump(proto_sents, fg, ensure_ascii=False, indent=4)
    print("Exported sentences to", args.outfile)


if __name__ == "__main__":
    main(create_args())

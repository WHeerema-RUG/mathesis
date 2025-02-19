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
import random


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
    parser.add_argument("-n", "--wordcount", type=int, default=100,
                        help="Amount of content words to generate"
                        "(default 100)")
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


def generate_syllable(onsets, nuclei, codas, tact="cVc"):
    """Generate a syllable using onset, nucleus and coda sets
    PHONOTACTICS: uppercase = required, lowercase = optional
    Only use C and V to define structure
    """
    # Set initial conditions
    onset = True
    syl = ""
    # Iterate through syllable pattern
    for i, seq in enumerate(tact):
        # If optional, use a function to determine dropout probability
        # TODO: Ground this distribution in literature
        if seq.lower() == seq and random.random() > 0.9 / (i + 1) ** (2 / 3):
            continue
        # If consonant, take from either onset or coda sets
        if seq.upper() == "C":
            if onset:
                syl = syl + random.choice(onsets)
            else:
                syl = syl + random.choice(codas)
        # If vowel, set it to coda
        elif seq.upper() == "V":
            syl = syl + random.choice(nuclei)
            onset = False
        else:
            raise ValueError("Invalid syllable shape")
    return syl


def generate_word(onsets, nuclei, codas, thres, d):
    """Generate a word based on the phonology and word lengths
    thres = word length thresholds
    d = divisor to get target syllable count
    The rest is fairly self-explanatory"""
    # Determine syllable count
    random_no = random.uniform(0.0, 0.99999999999)
    for key, value in thres.items():
        # If less than threshold, that is the target length
        if random_no < value:
            syl_count = round(int(key) / d)
            break
    # Generate syllables for the word
    word = ""
    for _ in range(syl_count):
        word = word + generate_syllable(onsets, nuclei, codas)
    return word


def generate_lexicon(phonology, feats, count, particles):
    """Separate function for lexicon generation to enhance code readability"""
    # Calculate the probability of it being a noun
    all_nouns = sum(feats["noun"].values())
    all_verbs = sum(feats["verb"].values())
    nounprob = all_nouns / (all_nouns + all_verbs)
    # Determine thresholds for word lengths
    # The lengths are the keys, the thresholds the values
    # in order to ease lookup
    thresholds = {"noun": {}, "verb": {}}
    # Nouns
    n_cache = 0.0
    for pair in sorted(feats["noun"].items(),
                      key=lambda item: item[1],
                      reverse=True):
        n_cache += pair[1] / all_nouns
        thresholds["noun"][pair[0]] = n_cache
    # Verbs
    v_cache = 0.0
    for pair in sorted(feats["verb"].items(),
                      key=lambda item: item[1],
                      reverse=True):
        v_cache += pair[1] / all_verbs
        thresholds["verb"][pair[0]] = v_cache
    # Determine available phonemes
    onsets = list(phonology["consonants"].keys())
    nuclei = list(phonology["vowels"])
    codas = []
    # Use sonority to determine the coda
    # Since sonorous phonemes occur more often in the coda,
    # there should be more opportunity to randomly select it
    # TODO: Sonority for bigger clusters
    for phoneme, s in phonology["consonants"].items():
        for _ in range(s+1):
            codas.append(phoneme)
    # Iterate to generate words
    lexicon = {"grammar": {}, "noun": [], "verb": []}
    # First the grammar
    for particle in particles:
        morpheme = generate_syllable(onsets, nuclei, codas, tact="CV")
        lexicon["grammar"][morpheme] = particle
    # Then the words
    for _ in range(count):
        # Determine noun or verb
        if random.random() < nounprob:
            word_type = "noun"
        else:
            word_type = "verb"
        # Generate a word, finally!
        word = generate_word(codas, nuclei, codas, thresholds[word_type], 3)
        lexicon[word_type].append(word)
    # Return finished lexicon
    return lexicon


def main(args):
    # Load or generate feature occurrences
    if args.featimport != "none":
        # If provided, load it instead of generating
        with open(args.featimport, "r") as ff:
            feat_occs = json.load(ff)
        # Disentangle feature set
        word_lens = {"noun": feat_occs.pop("noun"),
                     "verb": feat_occs.pop("verb")}
        print("Imported features from", args.featimport)
    else:
        # Else, generate it
        word_lens, feat_occs = feature_tally(brown.tagged_words())
        print("Tallied features")
        # If specified to export, do so
        if args.featexport != "none":
            # Thanks to phihag and Mateen Ulhaq on StackOverflow
            with open(args.featexport, "w", encoding="utf-8") as ff:
                json.dump(dict(feat_occs, **word_lens), ff,
                          ensure_ascii=False, indent=4)
            print("Exported features to", args.featexport)
    # Load in phonology
    with open(args.infile, "r") as fp:
        phonology = json.load(fp)
    print("Imported phonology from", args.infile)
    # Generate and export lexicon
    lexicon = generate_lexicon(phonology, word_lens, args.wordcount,
                               feat_occs.keys())
    with open(args.outfile, "w", encoding="utf-8") as fl:
        json.dump(lexicon, fl, ensure_ascii=False, indent=4)
    print("Exported lexicon to", args.outfile)


if __name__ == "__main__":
    main(create_args())

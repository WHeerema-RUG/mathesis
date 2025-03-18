# Do a morphology run using 10.000 sentences from 1000 words, 4000 BPE merges and 5 epochs
# Monograph orthography
python3 main.py sentences/gloss_10k.json sentences/agglutinative.json lexicon/lexicon-1k.json -s orth-mono.json -m 40000 -e 5 -o results_10k.csv
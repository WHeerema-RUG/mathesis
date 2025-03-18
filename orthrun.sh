# Do an orthography run using 10.000 sentences from 1000 words, 4000 BPE merges and 5 epochs
# Full agglutinative grammar
python3 main.py sentences/gloss_10k.json sentences/agglutinative.json lexicon/lexicon-1k.json -m 10000 -e 5 -o results-ortho_10k.csv
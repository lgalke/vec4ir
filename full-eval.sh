#!/bin/sh
OUT='glove20'
# MODEL='-m /data3/lgalke/GoogleNews-vectors-negative300.bin.gz'
MODEL='-m /data3/lgalke/glove.840B.300d.w2v_format.txt'
K=20

mkdir -p $OUT

# R1 titles
python3 ir_eval.py -k $K $MODEL -r1 -f title -t title -o "$OUT/r1-title-short.txt"
python3 ir_eval.py -k $K $MODEL -r1 -f title -t description -o "$OUT/r1-title-long.txt"
# R1 abstracts
python3 ir_eval.py -k $K $MODEL -r1 -f content -t title -o "$OUT/r1-abstract-short.txt"
python3 ir_eval.py -k $K $MODEL -r1 -f content -t description -o "$OUT/r1-abstract-long.txt"
# R2 titles
python3 ir_eval.py -k $K $MODEL -r2 -f title -t title -o "$OUT/r2-title-short.txt"
python3 ir_eval.py -k $K $MODEL -r2 -f title -t description -o "$OUT/r2-title-long.txt"
# R2 abstracts
python3 ir_eval.py -k $K $MODEL -r2 -f content -t title -o "$OUT/r2-abstract-short.txt"
python3 ir_eval.py -k $K $MODEL -r2 -f content -t description -o "$OUT/r2-abstract-long.txt"

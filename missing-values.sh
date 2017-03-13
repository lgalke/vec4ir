# Titles

## WCD with GLV and W2V

python3 ir_eval.py -d econ62k -ne word2vec -r wcd -I -k20 -j4 -o econ_missingno/word2vec-IWCD.tex
python3 ir_eval.py -d econ62k -ne glove -r wcd -I -k20 -j4 -o econ_missingno/glove-IWCD.tex

python3 ir_eval.py -d econ62k -ne word2vec -r wcd -k20 -j4 -o econ_missingno/word2vec-WCD.tex
python3 ir_eval.py -d econ62k -ne glove -r wcd -k20 -j4 -o econ_missingno/glove-WCD.tex

## WCMD with GLV and W2V
python3 ir_eval.py -d econ62k -ne word2vec -r wmd -k20 -j4 -o econ_missingno/word2vec-WMD.tex
python3 ir_eval.py -d econ62k -ne glove -r wmd -k20 -j4 -o econ_missingno/glove-WMD.tex

## D2V with D2V
python3 ir_eval.py -d econ62k -e doc2vec -r d2v -k20 -j4 -o econ_missingno/doc2vec.tex

# Full-text
python3 ir_eval.py -d econfull -e doc2vec -r d2v -k20 -j4 -o econ_missingno/doc2vec.tex

python3 ir_eval.py -d econfull -ne word2vec -r wcd -I -k20 -j4 -o econ_missingno/word2vec-IWCD.tex
python3 ir_eval.py -d econfull -ne glove -r wcd -I -k20 -j4 -o econ_missingno/glove-IWCD.tex

python3 ir_eval.py -d econfull -ne word2vec -r wcd -k20 -j4 -o econ_missingno/word2vec-WCD.tex
python3 ir_eval.py -d econfull -ne glove -r wcd -k20 -j4 -o econ_missingno/glove-WCD.tex

python3 ir_eval.py -d econfull -ne word2vec -r wmd -k20 -j4 -o econ_missingno/word2vec-WMD.tex
python3 ir_eval.py -d econfull -ne glove -r wmd -k20 -j4 -o econ_missingno/glove-WMD.tex

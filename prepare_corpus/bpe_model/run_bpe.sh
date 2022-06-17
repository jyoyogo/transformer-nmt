#git clone https://github.com/kh-kim/subword-nmt.git
DATA=/opt/project/translation/transformer-nmt/pretokenized_corpus
BPE_DATA=/opt/project/translation/transformer-nmt/bpe_corpus
python subword-nmt/learn_bpe.py --input ${DATA}/corpus_sample.train.tok.ko --output bpe.ko.model --symbols 30000 --verbose
python subword-nmt/learn_bpe.py --input ${DATA}/corpus_sample.train.tok.en --output bpe.en.model --symbols 50000 --verbose
head -n 5 bpe.*.model
echo "$(printf "\n\n")"
head -n 5 ${DATA}/corpus_sample.train.tok.ko
echo "$(printf "\n\n")"
head -n 5 ${DATA}/corpus_sample.train.tok.ko | python3 subword-nmt/apply_bpe.py -c bpe.ko.model
echo "$(printf "\n\n")"
head -n 5 ${DATA}/corpus_sample.train.tok.en
echo "$(printf "\n\n")"
head -n 5 ${DATA}/corpus_sample.train.tok.en | python3 subword-nmt/apply_bpe.py -c bpe.en.model
echo "$(printf "\n\n")"
cat ${DATA}/corpus_sample.train.tok.ko | python3 subword-nmt/apply_bpe.py -c bpe.ko.model > ${BPE_DATA}/corpus_sample.train.tok.bpe.ko ; cat ${DATA}/corpus_sample.valid.tok.ko | python3 subword-nmt/apply_bpe.py -c bpe.ko.model > ${BPE_DATA}/corpus_sample.valid.tok.bpe.ko ; cat ${DATA}/corpus_sample.test.tok.ko | python3 subword-nmt/apply_bpe.py -c bpe.ko.model > ${BPE_DATA}/corpus_sample.test.tok.bpe.ko
wc -l ${DATA}/corpus_sample.*.tok.ko
wc -l ${BPE_DATA}/corpus_sample.*.tok.bpe.ko
echo "$(printf "\n\n")"
cat ${DATA}/corpus_sample.train.tok.en | python3 subword-nmt/apply_bpe.py -c bpe.en.model > ${BPE_DATA}/corpus_sample.train.tok.bpe.en ; cat ${DATA}/corpus_sample.valid.tok.en | python3 subword-nmt/apply_bpe.py -c bpe.en.model > ${BPE_DATA}/corpus_sample.valid.tok.bpe.en ; cat ${DATA}/corpus_sample.test.tok.en | python3 subword-nmt/apply_bpe.py -c bpe.en.model > ${BPE_DATA}/corpus_sample.test.tok.bpe.en
wc -l ${DATA}/corpus_sample.*.tok.en
wc -l ${BPE_DATA}/corpus_sample.*.tok.bpe.en
echo "$(printf "\n\n")"
head -n 5 ${BPE_DATA}/corpus_sample.*.tok.bpe.ko
head -n 5 ${BPE_DATA}/corpus_sample.*.tok.bpe.en

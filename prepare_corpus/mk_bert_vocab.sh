#!/bin/bash
rm -rf ko_corpus/
rm -rf pretokenized_ko_corpus/
mkdir ko_corpus
split -a 4 -l 5000 -d ../data/corpus_sample.train.ko ko_corpus/corpus_sample.train.ko_
split -a 4 -l 5000 -d ../data/corpus_sample.valid.ko ko_corpus/corpus_sample.valid.ko_
split -a 4 -l 5000 -d ../data/corpus_sample.test.ko ko_corpus/corpus_sample.test.ko_
python3 pretokenize.py --tagger mecab --input_dir ko_corpus --output_dir pretokenized_ko_corpus --num_processes 4
mkdir pretokenized_corpus
cat pretokenized_ko_corpus/*.train.ko_* > pretokenized_corpus/corpus_sample.train.tok.ko
cat pretokenized_ko_corpus/*.valid.ko_* > pretokenized_corpus/corpus_sample.valid.tok.ko
cat pretokenized_ko_corpus/*.test.ko_* > pretokenized_corpus/corpus_sample.test.tok.ko
rm -rf ko_corpus
rm -rf pretokenized_ko_corpus
python3 train_bertwordpiece.py --type 'src' --files 'pretokenized_corpus/*.train.tok.ko' --name 'ko.bert-wordpiece' --out 'vocab'

rm -rf en_corpus/
rm -rf pretokenized_en_corpus/
mkdir en_corpus
split -a 4 -l 5000 -d ../data/corpus_sample.train.en en_corpus/corpus_sample.train.en_
split -a 4 -l 5000 -d ../data/corpus_sample.valid.en en_corpus/corpus_sample.valid.en_
split -a 4 -l 5000 -d ../data/corpus_sample.test.en en_corpus/corpus_sample.test.en_
python3 pretokenize.py --tagger moses --input_dir en_corpus --output_dir pretokenized_en_corpus --num_processes 4
cat pretokenized_en_corpus/*.train.en_* > pretokenized_corpus/corpus_sample.train.tok.en
cat pretokenized_en_corpus/*.valid.en_* > pretokenized_corpus/corpus_sample.valid.tok.en
cat pretokenized_en_corpus/*.test.en_* > pretokenized_corpus/corpus_sample.test.tok.en
rm -rf en_corpus
rm -rf pretokenized_en_corpus
python3 train_bertwordpiece.py --type 'tgt' --files 'pretokenized_corpus/*.train.tok.en' --name 'en.bert-wordpiece' --out 'vocab'

wc -l pretokenized_corpus/*
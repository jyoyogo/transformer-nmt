#!/bin/bash
DATADIR=/opt/project/translation/transformer-nmt/data
SAVEDIR=/opt/project/translation/transformer-nmt/pretokenized_corpus
FN_HEAD=corpus_sample
KO_SPLITDIR=ko_split_corpus
KO_TOKDIR=pretokenized_ko_split_corpus
EN_SPLITDIR=en_split_corpus
EN_TOKDIR=pretokenized_en_split_corpus

rm -rf ${SAVEDIR}/${KO_SPLITDIR}/
rm -rf ${SAVEDIR}/${KO_TOKDIR}/
rm -rf ${SAVEDIR}/${EN_SPLITDIR}
rm -rf ${SAVEDIR}/${EN_TOKDIR}/
mkdir ${SAVEDIR}/${KO_SPLITDIR}
mkdir ${SAVEDIR}/${KO_TOKDIR}
mkdir ${SAVEDIR}/${EN_SPLITDIR}
mkdir ${SAVEDIR}/${EN_TOKDIR}

split -a 4 -l 5000 -d ${DATADIR}/${FN_HEAD}.train.ko ${SAVEDIR}/${KO_SPLITDIR}/${FN_HEAD}.train.ko_
split -a 4 -l 5000 -d ${DATADIR}/${FN_HEAD}.valid.ko ${SAVEDIR}/${KO_SPLITDIR}/${FN_HEAD}.valid.ko_
split -a 4 -l 5000 -d ${DATADIR}/${FN_HEAD}.test.ko ${SAVEDIR}/${KO_SPLITDIR}/${FN_HEAD}.test.ko_

split -a 4 -l 5000 -d ${DATADIR}/${FN_HEAD}.train.en ${SAVEDIR}/${EN_SPLITDIR}/${FN_HEAD}.train.en_
split -a 4 -l 5000 -d ${DATADIR}/${FN_HEAD}.valid.en ${SAVEDIR}/${EN_SPLITDIR}/${FN_HEAD}.valid.en_
split -a 4 -l 5000 -d ${DATADIR}/${FN_HEAD}.test.en ${SAVEDIR}/${EN_SPLITDIR}/${FN_HEAD}.test.en_

python3 pretokenize.py --tagger mecab --input_dir ${SAVEDIR}/${KO_SPLITDIR} --output_dir ${SAVEDIR}/${KO_TOKDIR} --num_processes 16
python3 pretokenize.py --tagger moses --input_dir ${SAVEDIR}/${EN_SPLITDIR} --output_dir ${SAVEDIR}/${EN_TOKDIR} --num_processes 16

cat ${SAVEDIR}/${KO_TOKDIR}/*.train.ko_* > ${SAVEDIR}/${FN_HEAD}.train.tok.ko
cat ${SAVEDIR}/${KO_TOKDIR}/*.valid.ko_* > ${SAVEDIR}/${FN_HEAD}.valid.tok.ko
cat ${SAVEDIR}/${KO_TOKDIR}/*.test.ko_* > ${SAVEDIR}/${FN_HEAD}.test.tok.ko

cat ${SAVEDIR}/${EN_TOKDIR}/*.train.en_* > ${SAVEDIR}/${FN_HEAD}.train.tok.en
cat ${SAVEDIR}/${EN_TOKDIR}/*.valid.en_* > ${SAVEDIR}/${FN_HEAD}.valid.tok.en
cat ${SAVEDIR}/${EN_TOKDIR}/*.test.en_* > ${SAVEDIR}/${FN_HEAD}.test.tok.en

rm -rf ${SAVEDIR}/${KO_SPLITDIR}
rm -rf ${SAVEDIR}/${KO_TOKDIR}
rm -rf ${SAVEDIR}/${EN_SPLITDIR}
rm -rf ${SAVEDIR}/${EN_TOKDIR}

wc -l ${SAVEDIR}/${DESTDIR}/*

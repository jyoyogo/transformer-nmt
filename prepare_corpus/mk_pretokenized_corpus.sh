#!/bin/bash
DATADIR=/opt/project/translation/transformer-nmt/data
SAVEDIR=/opt/project/translation/transformer-nmt/pretokenized_corpus
SPLITDIR=ko_split_corpus
TOKDIR=pretokenized_ko_split_corpus

rm -rf ${SAVEDIR}/${SPLITDIR}/
rm -rf ${SAVEDIR}/${TOKDIR}/
mkdir ${SAVEDIR}/${SPLITDIR}
mkdir ${SAVEDIR}/${TOKDIR}

split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.train.ko ${SAVEDIR}/${SPLITDIR}/corpus_sample.train.ko_
split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.valid.ko ${SAVEDIR}/${SPLITDIR}/corpus_sample.valid.ko_
split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.test.ko ${SAVEDIR}/${SPLITDIR}/corpus_sample.test.ko_

python3 pretokenize.py --tagger mecab --input_dir ${SAVEDIR}/${SPLITDIR} --output_dir ${SAVEDIR}/${TOKDIR} --num_processes 4

cat ${SAVEDIR}/${TOKDIR}/*.train.ko_* > ${SAVEDIR}/corpus_sample.train.tok.ko
cat ${SAVEDIR}/${TOKDIR}/*.valid.ko_* > ${SAVEDIR}/corpus_sample.valid.tok.ko
cat ${SAVEDIR}/${TOKDIR}/*.test.ko_* > ${SAVEDIR}/corpus_sample.test.tok.ko
rm -rf ${SAVEDIR}/${SPLITDIR}
rm -rf ${SAVEDIR}/${TOKDIR}

wc -l ${SAVEDIR}/${DESTDIR}/*

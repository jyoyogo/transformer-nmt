#!/bin/bash
DATADIR=/opt/project/translation/transformer-nmt/data
SAVEDIR=/opt/project/translation/transformer-nmt/pretokenized_corpus
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

split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.train.ko ${SAVEDIR}/${KO_SPLITDIR}/corpus_sample.train.ko_
split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.valid.ko ${SAVEDIR}/${KO_SPLITDIR}/corpus_sample.valid.ko_
split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.test.ko ${SAVEDIR}/${KO_SPLITDIR}/corpus_sample.test.ko_

split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.train.en ${SAVEDIR}/${EN_SPLITDIR}/corpus_sample.train.en_
split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.valid.en ${SAVEDIR}/${EN_SPLITDIR}/corpus_sample.valid.en_
split -a 4 -l 5000 -d ${DATADIR}/corpus_sample.test.en ${SAVEDIR}/${EN_SPLITDIR}/corpus_sample.test.en_

python3 pretokenize.py --tagger mecab --input_dir ${SAVEDIR}/${KO_SPLITDIR} --output_dir ${SAVEDIR}/${KO_TOKDIR} --num_processes 16
python3 pretokenize.py --tagger moses --input_dir ${SAVEDIR}/${EN_SPLITDIR} --output_dir ${SAVEDIR}/${EN_TOKDIR} --num_processes 16

cat ${SAVEDIR}/${KO_TOKDIR}/*.train.ko_* > ${SAVEDIR}/corpus_sample.train.tok.ko
cat ${SAVEDIR}/${KO_TOKDIR}/*.valid.ko_* > ${SAVEDIR}/corpus_sample.valid.tok.ko
cat ${SAVEDIR}/${KO_TOKDIR}/*.test.ko_* > ${SAVEDIR}/corpus_sample.test.tok.ko

cat ${SAVEDIR}/${EN_TOKDIR}/*.train.en_* > ${SAVEDIR}/corpus_sample.train.tok.en
cat ${SAVEDIR}/${EN_TOKDIR}/*.valid.en_* > ${SAVEDIR}/corpus_sample.valid.tok.en
cat ${SAVEDIR}/${EN_TOKDIR}/*.test.en_* > ${SAVEDIR}/corpus_sample.test.tok.en

rm -rf ${SAVEDIR}/${KO_SPLITDIR}
rm -rf ${SAVEDIR}/${KO_TOKDIR}
rm -rf ${SAVEDIR}/${EN_SPLITDIR}
rm -rf ${SAVEDIR}/${EN_TOKDIR}

wc -l ${SAVEDIR}/${DESTDIR}/*

#!/bin/bash
python3 train_bertwordpiece.py --type 'src' --files 'pretokenized_corpus/*.train.tok.ko' --name 'ko.bert-wordpiece' --out 'vocab'
python3 train_bertwordpiece.py --type 'tgt' --files 'pretokenized_corpus/*.train.tok.en' --name 'en.bert-wordpiece' --out 'vocab'
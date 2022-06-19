import argparse
import os
import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument(
        "--path",
        default="/opt/project/translation/transformer-nmt/pretokenized_corpus",
        type=str,
    )
parser.add_argument(
        "--file",
        default="corpus_sample.train.tok",
        type=str,
        )
args = parser.parse_args()
ko_data_path = os.path.join(args.path, f"{args.file}.ko")
en_data_path = os.path.join(args.path, f"{args.file}.en")
spm.SentencePieceTrainer.Train(f'--input={ko_data_path} --model_prefix=ko.bpe --vocab_size=30000 --model_type=bpe --max_sentence_length=9999')
spm.SentencePieceTrainer.Train(f'--input={en_data_path} --model_prefix=en.bpe --vocab_size=50000 --model_type=bpe --max_sentence_length=9999')

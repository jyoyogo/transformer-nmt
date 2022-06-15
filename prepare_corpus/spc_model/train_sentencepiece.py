import sentencepiece as spm

ko_data_path = '/opt/project/translation/transformer-nmt/pretokenized_corpus/corpus_sample.train.tok.ko'
spm.SentencePieceTrainer.Train(f'--input={ko_data_path} --model_prefix=ko.bpe --vocab_size=30000 --model_type=bpe --max_sentence_length=9999')
en_data_path = '/opt/project/translation/transformer-nmt/pretokenized_corpus/corpus_sample.train.tok.en'
spm.SentencePieceTrainer.Train(f'--input={en_data_path} --model_prefix=en.bpe --vocab_size=50000 --model_type=bpe --max_sentence_length=9999')

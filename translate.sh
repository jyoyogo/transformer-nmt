echo "$(printf "**ref(test) source**")"
head -n 5 data/corpus.sample.test.tok.bpe.en | python apply_nmt/detokenizer.py
echo "$(printf "\n\n")"
echo "$(printf "**ref(test) target**")"
head -n 5 data/corpus.sample.test.tok.bpe.ko | python apply_nmt/detokenizer.py
echo "$(printf "\n\n")"
echo "$(printf "**inference result, beam_size 5**")"
head -n 5 data/corpus.sample.test.tok.bpe.en | python translate.py --model_fn checkpoint/nmt_model.30.*.pth --gpu_id -1 --batch_size 2 --beam_size 1 | python apply_nmt/detokenizer.py

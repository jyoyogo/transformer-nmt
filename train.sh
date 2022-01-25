#!/bin/bash
python train.py --train ./data/sample.train --valid ./data/sample.valid --lang enko \
--gpu_id 1,2 --batch_size 128 --n_epochs 1 --verbose 2 --max_length 100 --dropout .2 \
--hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 32 \
--lr 1e-3 --lr_step 0 --use_adam --use_transformer --rl_n_epochs 0 \
--model_fn ./checkpoint/nmt_model.pth

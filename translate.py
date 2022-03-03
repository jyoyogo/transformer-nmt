import argparse
import sys
import codecs
from operator import itemgetter

import torch
from torch.nn.utils.rnn import pad_sequence
from model.transformers import Transformer

from data_loader.nmt_loader import NmtDataLoader
import data_loader.nmt_loader as data_loader

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default=None,
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter results. Default=%(default)s',
    )

    config = p.parse_args()

    return config

def read_text(batch_size=128):
    # This method gets sentences from standard input and tokenize those.
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= batch_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines

def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []
    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                print('EOS!!!')
                break
            else:
                line += [vocab.itos[int(index)]]

        line = ' '.join(line)
        lines += [line]

    return lines

def get_vocabs(saved_data):

    # Load vocabularies from the model.
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab

def get_model(input_size, output_size, train_config, saved_data):
    # Declare sequence-to-sequence model.
    if 'use_transformer' in vars(train_config).keys() and train_config.use_transformer:
        model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_splits=train_config.n_splits,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
    else:
        pass

    model.load_state_dict(saved_data['model'])  # Load weight parameters from the trained model.
    model.eval()  # We need to turn-on the evaluation mode, which turns off all drop-outs.

    return model


if __name__ == '__main__':
    # sys.argv = ['translate.py', '--model_fn', '/home/user/transformer-nmt/checkpoint/nmt_model.30.1.83-6.20.3.08-21.79.pth', '--gpu_id', '-1', '--batch_size', '2', '--gpu_id', '-1', '--batch_size', '2', '--beam_size', '1']
    config = define_argparser()

    # Load saved model.
    saved_data = torch.load(
        config.model_fn
    )

    # Load configuration setting in training.
    train_config = saved_data['config']

    src_vocab, tgt_vocab = get_vocabs(saved_data)
    
    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    loader = NmtDataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)

    input_size, output_size = len(loader.src_vocab), len(loader.tgt_vocab)
    model = get_model(input_size, output_size, train_config, saved_data)

    # Put models to device if it is necessary.
    device = 'cuda:%s' % config.gpu_id if int(config.gpu_id) >= 0 else 'cpu'

    if config.gpu_id >= 0:
        model.cuda(device)
    
    with torch.no_grad():
        # Get sentences from standard input.
        for lines in read_text(batch_size=config.batch_size):
            
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            lengths         = [len(line) for line in lines]
            original_indice = [i for i in range(len(lines))]

            sorted_tuples = sorted(
                zip(lines, lengths, original_indice),
                key=itemgetter(1),
                reverse=True,
            )
            sorted_lines    = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths         = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]
            
            numericalized_text = [data_loader.numericalize(s, src_vocab.stoi, device) for s in sorted_lines]
            x = data_loader.padding_batch(numericalized_text)
            
            # Converts string to list of index.
            # x = loader.src.numericalize(
            #     loader.src.pad(sorted_lines),
            #     device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            # )
            # |x| = (batch, sentence_length)

            if config.beam_size == 1:
                y_hats, indice = model.search(x)
                # |y_hats| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)

                output = to_text(indice, loader.tgt_vocab)
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            
            else:
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty,
                )

                # Restore the original_indice.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt_vocab)]
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
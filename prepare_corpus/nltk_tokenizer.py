# -*- coding: utf-8 -*-
import time
import sys, fileinput
import argparse
from nltk.tokenize import word_tokenize
from transformers import BasicTokenizer

class EnTokenizer(object):
    def __init__(self):
        # Split punctuation & Tokenize Chinese Character & Clean Text
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, tokenize_chinese_chars=True)

    def tokenize(self, raw_text: str):
        # print(raw_text)
        sent = " ".join(self.basic_tokenizer.tokenize(raw_text))
        buf = ['▁']
        ref_idx = 0
        tok_idx = 0

        pretokenized_text = ' '.join(word_tokenize(raw_text.strip()))

        while tok_idx < len(pretokenized_text):
            c = pretokenized_text[tok_idx]
            
            tok_idx += 1

            if c != ' ':
                while ref_idx < len(raw_text):
                    c_ = raw_text[ref_idx]
                    
                    ref_idx += 1

                    if c_ == ' ':
                        c = '▁' + c
                    else:
                        break
            buf += [c]

        return ''.join(buf)
        '''
        text_ptr = 0
        is_first_token = True
        tokenized = []
        pretokenized = word_tokenize(sent.strip())
        for morph in pretokenized:
            token = morph

            # If space token, increment text_ptr
            # try:
            if sent[text_ptr] == " ":
                while sent[text_ptr] == " ":
                    text_ptr += 1
                is_first_token = True  # Reset that it is first token

            text_ptr += len(token)

            if is_first_token:
                is_first_token = False
            else:
                token = "##" + token
                pass

            tokenized.append(token)

        return tokenized
        '''
if __name__ == "__main__":
    tokenizer = EnTokenizer()
    start = time.time()
    text = "How can Company A and B get relief for damages caused by A's breach of obligations?"
    text = "Don't act like a fool in front of people ."
    print(''.join(tokenizer.tokenize(text)))
    end = time.time()
    print(end - start)
    ''' 
    for line in fileinput.input():
        if line.strip() != "":
            tokens = tokenizer.tokenize(line.strip())
            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')
    '''

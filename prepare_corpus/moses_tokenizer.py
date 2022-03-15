# -*- coding: utf-8 -*-
import sys, fileinput
import argparse
from mosestokenizer import *
from transformers import BasicTokenizer

class EnTokenizer(object):
    def __init__(self):
        # Split punctuation & Tokenize Chinese Character & Clean Text
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, tokenize_chinese_chars=True)

    def tokenize(self, text: str):
        text = " ".join(self.basic_tokenizer.tokenize(text))
        text_ptr = 0
        is_first_token = True
        tokenized = []
        with MosesTokenizer('en') as tokenize:
            # tokenized_text = ' '.join(tokenize(text.strip())).replace("&quot;", '"').replace("&apos;","'").replace("&lt;","<").replace("'&gt;'",">")
            for token in tokenize(text.strip()):
            # for token in tokenized_text.split(' '):
                # If space token, increment text_ptr
                # try:
                if text[text_ptr] == " ":
                    while text[text_ptr] == " ":
                        text_ptr += 1
                    is_first_token = True  # Reset that it is first token
                
                if token == text[text_ptr:text_ptr+len(token)]:
                    pass
                else:
                    #cover html and xml special char, as like &quot;, &apos; , etc...
                    token = text[text_ptr:text_ptr+len(token)].split(' ')[0]

                text_ptr += len(token)

                if is_first_token:
                    token = "_" + token
                    is_first_token = False
                else:
                    # token = "##" + token
                    pass

                tokenized.append(token)

        return tokenized

if __name__ == "__main__":
    tokenizer = EnTokenizer()
    # text = "How can Company A and B get relief for damages caused by A's breach of obligations?"
    # print(tokenizer.tokenize(text))
    for line in fileinput.input():
        if line.strip() != "":
            tokens = tokenizer.tokenize(line.strip())
            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')

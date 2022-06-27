import sys, fileinput
import MeCab
from transformers import BasicTokenizer


class MecabTokenizer(object):
    def __init__(self):
        self.mecab = MeCab.Tagger(f"-r /dev/null -d /opt/project/translation/transformer-nmt/transformer_env/lib/mecab/dic/mecab-ko-dic")
        # Split punctuation & Tokenize Chinese Character & Clean Text
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, tokenize_chinese_chars=True)

    def tokenize(self, raw_text: str):
        sent = " ".join(self.basic_tokenizer.tokenize(raw_text))
        buf = ['▁']
        ref_idx = 0
        tok_idx = 0

        pretokenized_text = ' '.join([m.split('\t')[0] for m in self.mecab.parse(sent).split("\n") if '\t' in m])

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

        return buf

        # for morph in pretokenized:
        
        #     token = morph
            
        #     # If space token, increment text_ptr
        #     try:
        #         if sent[text_ptr] == " ":
        #             while sent[text_ptr] == " ":
        #                 text_ptr += 1
                    
        #             is_first_token = True  # Reset that it is first token
        #     except Exception as ex :
        #         print(ex)

        #     text_ptr += len(token)

        #     if is_first_token:
        #         is_first_token = False
        #         print(text_ptr,)
        #     else:
        #         token = "##" + token
        #         print(token)
        #         #pass
            
        #     tokenized.append(token)
        # return tokenized


if __name__ == "__main__":
    tokenizer = MecabTokenizer()
    text = "본 OKC 페이퍼에는, 이번 emnlp finding's에 억셉된 두 개의 한국어 nlp dataset들이 수록되어 있습니다. 左衝右突"
    print(''.join(tokenizer.tokenize(text)).replace(' ', '').replace('▁', ' ').strip())
    '''
    for line in fileinput.input():
        if line.strip() != "":
            tokens = tokenizer.tokenize(line.strip())
            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')
    '''

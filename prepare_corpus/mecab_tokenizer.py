import sys, fileinput
import MeCab
from transformers import BasicTokenizer


class MecabTokenizer(object):
    def __init__(self):
        self.mecab = MeCab.Tagger(f"-r /dev/null -d /home/user/myenv/lib/mecab/dic/mecab-ko-dic")
        # Split punctuation & Tokenize Chinese Character & Clean Text
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False, tokenize_chinese_chars=True)

    def tokenize(self, text: str):
        text = " ".join(self.basic_tokenizer.tokenize(text))
        text_ptr = 0
        is_first_token = True
        tokenized = []

        for morph in self.mecab.parse(text).split("\n"):
            if "\t" in morph:
                token = morph.split("\t")[0]

                # If space token, increment text_ptr
                if text[text_ptr] == " ":
                    while text[text_ptr] == " ":
                        text_ptr += 1
                    is_first_token = True  # Reset that it is first token

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
    tokenizer = MecabTokenizer()
    # text = "본 OKC 페이퍼에는, 이번 emnlp findings에 억셉된 두 개의 한국어 nlp dataset들이 수록되어 있습니다. 左衝右突"
    # print(' '.join(tokenizer.tokenize(text)))
    for line in fileinput.input():
        if line.strip() != "":
            tokens = tokenizer.tokenize(line.strip())
            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')

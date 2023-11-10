
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import os

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    # 测试分词器
    tokenizer = CLIPTokenizer(vocab_file='./sd2.1-tokenizer/vocab.json', merges_file='./sd2.1-tokenizer/merges.txt')
    r = tokenizer.encode('A horse with a dog')
    print(r)

import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))

from transformers import BertJapaneseTokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
from transformers import BertForMaskedLM
net = BertForMaskedLM.from_pretrained('bert-base-japanese-whole-word-masking')

tokenizer.save_pretrained(os.path.join(ROOT,"model","tokenizer"))
net.save_pretrained(os.path.join(ROOT,"model"))
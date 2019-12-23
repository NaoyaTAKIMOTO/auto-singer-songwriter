from transformers.tokenization_bert_japanese import BertJapaneseTokenizer,MecabTokenizer
from transformers.modeling_bert import BertForMaskedLM

tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
model = BertForMaskedLM.from_pretrained('bert-base-japanese-whole-word-masking')

import os

ROOT=os.path.dirname(__file__)
model_path = os.path.join(ROOT,"model")
tokenizer_path = os.path.join(ROOT,"model","tokenizer")

tokenizer.save_pretrained(tokenizer_path)
model.save_pretrained(model_path)
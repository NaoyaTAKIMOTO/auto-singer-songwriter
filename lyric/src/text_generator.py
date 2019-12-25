import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import random
from transformers import BertJapaneseTokenizer
from transformers import BertForMaskedLM

tokenizer = BertJapaneseTokenizer.from_pretrained(os.path.join(ROOT, "model", "tokenizer"))
model = BertForMaskedLM.from_pretrained(os.path.join(ROOT, "model"))
class Node(object):
    def __init__(self,input_ids ,score):
        self.input_ids=input_ids
        self.score=score

def text_generate(input_text, beam_width=5, is_random_insert=True ,is_add=True, add_depth=3):
    mask_id = tokenizer.mask_token_id
    text = input_text
    input_ids = tokenizer.encode(text, return_tensors='pt')
    #print(input_ids)
    input_ids = input_ids.tolist()[0]
    #print(input_ids)


    if is_random_insert:
        for _ in range(5):
            random_ind=random.randrange(len(input_ids))
            input_ids.insert(random_ind,mask_id)

    if is_add:
        for _ in range(add_depth):
            input_ids.insert(-2, mask_id)

    input_ids=torch.tensor(input_ids).view(1,-1)

    nodes = [Node(input_ids, 1)]

    masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()
    #print("mask ind:{}".format(masked_indices))
    for masked_index in masked_indices:
        #print("mask index:{}".format(masked_index))
        new_nodes = []
        for node in nodes:
            input_ids = node.input_ids
            result = model(input_ids)
            pred_ids = result[0][:, masked_index].topk(5).indices.tolist()[0]
            pred_scores = result[0][:, masked_index].topk(5).values.tolist()[0]
            #remove repeat token
            previous_token_id = input_ids.tolist()[0][masked_index-1]
            if previous_token_id in pred_ids:
                pred_ids.remove(previous_token_id)

            for j, pred_id in enumerate(pred_ids):
                output_ids = input_ids.tolist()[0]
                output_ids[masked_index] = pred_id
                # print(tokenizer.decode(output_ids))
                new_nodes.append(Node(torch.tensor(output_ids).view(1, -1), node.score * pred_scores[j]))
        # prune
        if len(new_nodes) > beam_width:
            scores = []
            for node in new_nodes:
                scores.append(node.score)

            descending_order = np.argsort(np.asarray(scores))[::-1]
            del nodes
            nodes = []
            for j in range(beam_width):
                nodes.append(new_nodes[descending_order[j]])
        else:
            nodes = new_nodes

    dict={}
    for i,node in enumerate(nodes):
        dict[i]=[tokenizer.decode(node.input_ids.tolist()[0]), node.score]

    return dict

if __name__ == '__main__':

    text_generate("青葉山でこんなことをしています")


    beam_width=5

    tokenizer = BertJapaneseTokenizer.from_pretrained(os.path.join(ROOT,"model","tokenizer"))
    model= BertForMaskedLM.from_pretrained(os.path.join(ROOT,"model"))

    text = f'''
        青葉山で{tokenizer.mask_token}の{tokenizer.mask_token}をして{tokenizer.mask_token}います。
    '''
    input_ids = tokenizer.encode(text, return_tensors='pt')
    print(input_ids)

    nodes = [Node(input_ids ,1) ]

    masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()
    print("mask ind:{}".format(masked_indices))
    for masked_index in masked_indices:
        print("mask index:{}".format(masked_index))
        new_nodes=[]
        for node in nodes:
            input_ids=node.input_ids
            result = model(input_ids)
            pred_ids = result[0][:, masked_index].topk(5).indices.tolist()[0]
            pred_scores = result[0][:, masked_index].topk(5).values.tolist()[0]
            for j, pred_id in enumerate(pred_ids):
                output_ids = input_ids.tolist()[0]
                output_ids[masked_index] = pred_id
                #print(tokenizer.decode(output_ids))
                new_nodes.append(Node(torch.tensor(output_ids).view(1,-1), node.score*pred_scores[j]))
        #prune
        if len(new_nodes)>beam_width:
            scores =[]
            for node in new_nodes:
                scores.append(node.score)

            descending_order=np.argsort(np.asarray(scores))[::-1]
            del nodes
            nodes=[]
            for j in range(beam_width):
                nodes.append(new_nodes[descending_order[j]])
        else:
            nodes=new_nodes

    for node in nodes:
        print (tokenizer.decode(node.input_ids.tolist()[0]))

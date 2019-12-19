import torch
import torch.nn as nn
import numpy as np
import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer,MecabTokenizer
from transformers.modeling_bert import BertForMaskedLM

tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
model = BertForMaskedLM.from_pretrained('bert-base-japanese-whole-word-masking')


class BeamNodes(object):
    def __init__(self,prob, ids):
        self.prob=prob
        self.ids=ids


class BeamDecoder(object):
    """

    example:
    bead_decoder=BeamDecoder(model,tokenizer)
    input_ids=bead_decoder.encode("明日が",num_mask=4)
    bead_decoder.decode(input_ids)
    """
    def __init__(self, model, tokenizer, beam_width=5):
        self.model = model
        self.softmax = nn.Softmax(-1)
        self.beam_width = beam_width

    def encode(self, lyric, num_mask=5):
        for _ in range(num_mask):
            lyric = lyric + tokenizer.mask_token
        input_ids = tokenizer.encode(lyric, return_tensors='pt')
        return input_ids

    def decode(self, input_ids):

        self.masked_indexes = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()
        self.result = self.model(input_ids)
        self.pred_ids = self.result[0][:, self.masked_indexes].topk(10).indices
        self.probabilities = softmax(self.result[0][:, self.masked_indexes].topk(10).values)
        self.parent_nodes = [BeamNodes(1, input_ids)]

        for j, masked_index in enumerate(self.masked_indexes):
            if j == 0:
                prob_parent = 1
            childre_nodes = []
            for i in range(len(self.parent_nodes)):
                parent_node = self.parent_nodes.pop()
                input_ids = parent_node.ids.clone().detach()
                result = self.model(input_ids)
                pred_ids = result[0][:, self.masked_indexes].topk(10).indices

                probabilities = softmax(result[0][:, self.masked_indexes].topk(10).values)
                probabilities = torch.where(pred_ids[0] != input_ids[:, masked_index - 1], probabilities,
                                            torch.zeros_like(probabilities))
                for k in range(self.beam_width):
                    output_ids = parent_node.ids.clone().detach()
                    output_ids[:, masked_index] = pred_ids[:, j, k]
                    child_node = BeamNodes(prob_parent * probabilities[:, j, k], output_ids)
                    childre_nodes.append(child_node)
                    # print(tokenizer.decode(output_ids.tolist()[0]))

            # prune
            current_probs_order = np.argsort(np.asarray([node.prob for node in childre_nodes]), axis=0)[::-1]
            self.parent_nodes = np.asarray(childre_nodes)[current_probs_order[:beam_width]].tolist()
            # print("")

        n_best=[]
        for node in self.parent_nodes:
            output_ids = node.ids.clone().detach()
            #print(tokenizer.decode(output_ids.tolist()[0]))
            n_best.append(tokenizer.decode(output_ids.tolist()[0]))


        return n_best


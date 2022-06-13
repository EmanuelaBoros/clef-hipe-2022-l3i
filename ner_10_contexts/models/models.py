# -*- coding: utf-8 -*-

from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder, MultiHeadAttn, TransformerLayer
from torch import nn
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from fastNLP.core.const import Const as C
from fastNLP.modules.encoder.lstm import LSTM
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules.decoder.mlp import MLP


class StackedTransformersCRF(nn.Module):
    def __init__(self, tag_vocabs, embed, embed_doc, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):

        super().__init__()

        self.embed = embed
        self.embed_doc = embed_doc

        num_layers = 1
 
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        self.tag_vocabs = []
        self.out_fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()
        
        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            
#            linear = nn.Linear(768, len(tag_vocabs[i]))
            linear = nn.Linear(1536, len(tag_vocabs[i]))
#            linear = nn.Linear(9216, len(tag_vocabs[i]))
            
#            linear = nn.Linear(1792, len(tag_vocabs[i]))
            self.out_fcs.append(linear)
            
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type='bioes', include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)
            
        
        self.in_fc = nn.Linear(4608, d_model)
        

        self.in_fc_docs = nn.ModuleList() 
        for i in range(11):
            self.in_fc_docs.append(nn.Linear(4608, d_model))
#        self.in_fc_doc_flip = nn.Linear(d_model, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        #self.transformer.requires_grad = True
#        n_heads = 12# 
#        head_dims = 128
        self.transformer_doc = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        
        self.self_attn = MultiHeadAttn(d_model, n_head)
        
        hidden_dim = 512
#        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.lstm = LSTM(input_size=self.embed_doc.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        
        self.pooling_methods = ['max', 'mean', 'max-mean']
        
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc_dropout_doc = nn.Dropout(fc_dropout)
        
        
        self.linear_bottleneck1 = nn.Linear(1536, 6144)
        self.linear_bottleneck2 = nn.Linear(6144, 1536)
        
        adapter_size = 6144
        
        self.bottleneck = nn.Sequential(nn.Linear(d_model, adapter_size),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(adapter_size, d_model),
                                 nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        
    def _mean_pooler(self, encoding):
        return encoding.mean(dim=1)
    
    def _max_pooler(self, encoding):
        return encoding.max(dim=1).values
    
    def _max_mean_pooler(self, encoding):
        return torch.cat((self._max_pooler(encoding), self._mean_pooler(encoding)), dim=1)
    
    def _pooler(self, encodings, pooling_method):
        '''
        Pools the encodings along the time/sequence axis according
        to one of the pooling method:
            - 'max'      :  max value along the sequence/time dimension
                            returns a (batch_size x hidden_size) shaped tensor
            - 'mean'     :  mean of the values along the sequence/time dimension
                            returns a (batch_size x hidden_size) shaped tensor
            - 'max-mean' :  max and mean values along the sequence/time dimension appended
                            returns a (batch_size x 2*hidden_size) shaped tensor
                            [ max : mean ]
        Parameters
        ----------
        encoding : list of tensor to pool along the sequence/time dimension.
        
        pooling_method : one of 'max', 'mean' or 'max-mean'
        
        Returns
        -------
        tensor of shape (batch_size x hidden_size).
        '''
        
        assert (pooling_method in self.pooling_methods), \
            "pooling methods needs to be one of 'max', 'mean' or 'max-mean'"
            
        if pooling_method   == 'max':       pool_fn = self._max_pooler
        elif pooling_method == 'mean':      pool_fn = self._mean_pooler
        elif pooling_method == 'max-mean':  pool_fn = self._max_mean_pooler
        
        pooled = pool_fn(encodings)
        
        return pooled
    
    def _forward(self, words, doc=None, doc0=None, doc1=None, doc2=None, doc3=None, doc4=None, doc5=None, doc6=None, doc7=None, doc8=None, doc9=None, target=None, target1=None, target2=None, target3=None, 
                 target4=None, target5=None, target6=None, bigrams=None, seq_len=None):

        torch.cuda.empty_cache()

        #import pdb;pdb.set_trace()
        mask = words.ne(0)
        words = self.embed(words)
        
        words_doc = []
        masks_doc = []
        for x in [doc0, doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9]:
            #print(x)
            if x is None: x = doc
            mask_doc = x.ne(0)
            masks_doc.append(mask_doc)
            words_doc.append(self.embed(x))

        torch.cuda.empty_cache()
        targets = [target, target1, target2, target3, target4, target5]

        chars = self.in_fc(words)

        for idx, chars_doc in enumerate(words_doc):
            words_doc[idx] = self.in_fc_docs[idx](chars_doc)
        
#        print('doc', doc.shape)

        chars = self.transformer(chars, mask)
        
#        for idx, _ in enumerate(words_doc):
#            words_doc[idx] = self.fc_dropout(self.transformer(words_doc[idx], masks_doc[idx]))

        words = self.fc_dropout(chars)
#        print('words', words.shape)
        torch.cuda.empty_cache()
        
        jokers = []
        for idx, _ in enumerate(words_doc):

            joker = self._pooler(words_doc[idx], 'mean').unsqueeze(1)
            joker = joker.reshape(joker.shape[0], 1, -1)
            jokers.append(joker)

        #import pdb;pdb.set_trace()
        #joker = self.bottleneck(joker)
        #joker = self.norm(joker)
        #import pdb;pdb.set_trace()
        joker = torch.stack(jokers)
        #import pdb;pdb.set_trace()
        #words = torch.cat([words, joker], 1)
        
        words = torch.cat([words, joker.reshape(words.shape[0], joker.shape[0], -1)], 1)
        
        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(self.out_fcs[i](words), dim=-1))

        torch.cuda.empty_cache()
        
        #import pidb;pdb.set_trace()
        joker_mask = torch.ones((words.shape[0], 10)).bool().to('cuda')
        #import pdb;pdb.set_trace()
         
        joker_mask = torch.cat([mask, joker_mask], 1)
                
        if target is not None:
            losses = []
            for i in range(len(targets)):
            
                joker_targets = torch.zeros((targets[i].shape[0], 10)).to('cuda')
                joker_targets = torch.cat([targets[i], joker_targets], 1)

                losses.append(self.crfs[i](logits[i], joker_targets, joker_mask))
#                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                #import pdb;pdb.set_trace()
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], joker_mask)[0][:,:-10]
#                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                else:
                    results['pred' + str(i)] = torch.argmax(logits[i], 2)[:,:-10]
#                    results['pred' + str(i)] = torch.argmax(logits[i], 2)
                    # results['pred' + str(i)] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
#                import pdb;pdb.set_trace()
            return results

    def forward(self, words, doc=None, doc0=None, doc1=None, doc2=None, doc3=None, doc4=None, doc5=None, doc6=None, doc7=None, doc8=None, doc9=None, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, seq_len=None):
        return self._forward(words, doc, doc0, doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, target, target1, target2, target3, target4, target5, target6, seq_len)

    def predict(self, words, doc=None, seq_len=None):
        return self._forward(words, doc, target=None)


class BertCRF(nn.Module):
    def __init__(self, embed, tag_vocabs, encoding_type='bio'):
        super().__init__()
        self.embed = embed
        self.tag_vocabs = []
        self.fcs = nn.ModuleList()
        self.crfs = nn.ModuleList()

        for i in range(len(tag_vocabs)):
            self.tag_vocabs.append(tag_vocabs[i])
            linear = nn.Linear(self.embed.embed_size, len(tag_vocabs[i]))
            self.fcs.append(linear)
            trans = allowed_transitions(
                tag_vocabs[i], encoding_type=encoding_type, include_start_end=True)
            crf = ConditionalRandomField(
                len(tag_vocabs[i]), include_start_end_trans=True, allowed_transitions=trans)
            self.crfs.append(crf)

    def _forward(self, words, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, seq_len=None):
        mask = words.ne(0)
        words = self.embed(words)

        targets = [target]#, target1, target2, target3, target4, target5, target6]

        words_fcs = []
        for i in range(len(targets)):
            words_fcs.append(self.fcs[i](words))

        logits = []
        for i in range(len(targets)):
            logits.append(F.log_softmax(words_fcs[i], dim=-1))

        if target is not None:
            losses = []
            for i in range(len(targets)):
                losses.append(self.crfs[i](logits[i], targets[i], mask))

            return {'loss': sum(losses)}
        else:
            results = {}
            for i in range(len(targets)):
                if i == 0:
                    results['pred'] = self.crfs[i].viterbi_decode(logits[i], mask)[0]
                else:
                    results['pred' + str(i)] = torch.argmax(logits[i], 2)

            return results

    def forward(self, words, target=None, target1=None, target2=None, target3=None, target4=None, target5=None, target6=None, seq_len=None):
        return self._forward(words, target, target1, target2, target3, target4, target5, target6, seq_len)

    def predict(self, words, seq_len=None):
        return self._forward(words, target=None)

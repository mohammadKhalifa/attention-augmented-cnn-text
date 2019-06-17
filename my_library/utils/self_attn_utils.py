import torch
import torch.nn as nn
import numpy as np
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query = self.linears[0](query)
        key = self.linears[1](key)
        value = self.linears[2](value)

        # extracting heads
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(2,1)
        key = key.view(nbatches, -1, self.h, self.d_k).transpose(2,1)
        value = value.view(nbatches, -1, self.h, self.d_k).transpose(2,1)


        attn_output, self.attn = attention(query, key, value, mask=mask)
        #concatenating heads output
        return self.linears[-1](attn_output.view(nbatches, -1, self.h * self.d_k))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


if __name__=='__main__':

      #creating sample tokens tensor of shape (1, 5, 100)
    tokens = torch.FloatTensor(np.random.randint(1, 100, size=(5,100)))
    tokens = tokens.unsqueeze(0)
    
    PE = PositionalEncoding(100)
    tokens = PE(tokens)
    attn = MultiHeadedAttention(4, 100)

    attn_result = attn(tokens, tokens, tokens)
    
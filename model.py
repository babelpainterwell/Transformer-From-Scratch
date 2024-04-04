import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init()
        self.d_model = d_model 
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x): # [batch, seq_len]
        return self.embedding(x) * math.sqrt(self.d_model) # [batch, seq_len, d_model]


# requires_grad_(False) -- to make a particular tensor not learned

class PosionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout_rate: float):
        super().__init__()
        self.drop = nn.Dropout(p=dropout_rate)
        
        # PE(pos, 2i) = sin(pos/10000^{2i/d_model})
        # PE(pos, 2i+1) = cos(pos/10000^{2i/d_model})
        encoding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = 

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0) # add a batch dimension in the front 


        # register a buffer to avoid the encoding from being considered as a learnable parameter 
        self.register_buffer("pe", encoding)

    
    def forward(self, x):
        # add the positional encoding to the input sequence embedding [batch, seq_len, d_model]
        x = x + self.pe[:, :x.shape[1], :]    # .requires_grad_(False) unnecessary if self.pe has been registered as a buffer
        return self.dropout(x)
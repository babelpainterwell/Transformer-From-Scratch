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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0) / d_model))
        # div_term = 10000 ** (torch.arange(0, d_model,2) / d_model)
        # encoding[:, 0::2] = torch.sin(position / div_term)
        # encoding[:, 1::2] = torch.cos(position / div_term)

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0) # add a batch dimension in the front, [1, seq_len, d_model]

        # register a buffer to avoid the encoding from being considered as a learnable parameter 
        self.register_buffer("pe", encoding)

    
    def forward(self, x):
        # add the positional encoding to the input sequence embedding [batch, seq_len, d_model]
        # .requires_grad_(False) unnecessary if self.pe has been registered as a buffer
        # make sure we make the added positional encoding have the same length as the maximum input length
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps 
        # scaling and centering are learnable parameters
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied 
        self.bias = nn.Parameter(torch.zeros(1)) # added 

    def forward(self, x):
        # input shape: [batch, seq_length, d_model]
        # do not reduce the dimension, [batch, seq_len, 1]
        # The dimension to reduce is the last one, meaning keep the first two dimesnions [batch, seq_len]
        # For batch norm, we reduce the batch dimension, so we keep the last two dimensions
        mean = x.mean(dim = -1, keepdim=True) 
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.h = h 
        self.d_model = d_model
        assert self.d_model % self.h == 0, "Embedding dimension is not divisble by the number of heads."

        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        '''
        Static method allows a method to be called even without a class instance. eg. MultiHeadAttentionBlock.attention()
        return the product of attention score matrix and V, and the attention scores matrix itself

        query, key, value have shape of [batch, h, seq_len, d_k]
        '''
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k) # [batch, h, seq_len, d_k] --> [batch, h, seq_len, seq_len], so key has to be transposed to [batch, h, d_k, seq_len] by switch the last two dimensions
        
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores) # WHY DO WE NEED DROPOUT HERE?; attention_scores has shape of [batch, h, seq_len, seq_len]
        '''
        Applying dropout to these scores helps in preventing the model from relying too heavily on specific parts of the input for making predictions. 
        This is particularly important in complex models like those used in NLP, where the model might overfit to the training data by always paying 
        attention to the same inputs when generating outputs.
        '''

        return (attention_scores @ value), attention_scores



    # Be careful with the order
    def forward(self, q, k, v, mask): # Used for both the encoder and the decoder
        '''
        q, k, v are having the size of [batch, seq_len, d_model].
        need to turn them into [batch, h, seq_len, d_k] by turning them into [batch, seq_len, h, d_k] first. 
        '''
        
        query = self.w_q(q).view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2) # [batch, h, seq_len, d_k]
        key = self.w_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)  # [batch, h, seq_len, d_k]
        value = self.w_v(v).view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)  # [batch, h, seq_len, d_k]

        # Calculate the attention using query, key, value (multi-head version)
        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # x has shape [batch, h, seq_len, d_k]

        # concatenate multiple heads: [batch, h, seq_len, d_k] --> [batch, seq_len, d_model]
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], self.d_model)


        return x # [batch, seq_len, d_model]


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization()
        # sublayer?
    
    def forward(self, x, sublayer):
        # sublayer here is a function
        return x + self.dropout(sublayer(x.norm(x))) # we could also sublayer first and then norm it; sublayer could be the Feed Forward layer or the attention layer
        # PAPER VERSION:
        # return self.norm(x + self.dropout(sublayer(x))) 
        # what if the function needs multiple input such as MultiHeadAttentionBlock? q, k, v and mask? - use lambda, which only takes 1 argument as input x 
    


class EncoderBlock(nn.Module):
    """
    Two residual connections: one for multi-head attention, the other for feed forward 
    A src_mask is also needed in order to avoid the attention between words and padding words
    """

    # def __init__(self, N: int, d_model: int, d_ff: int, h: int, dropout:float):
    #     super().__init__()
    #     self.N = N
    #     self.attentionResidual = ResidualConnection(dropout)
    #     self.feedForwardResidual = ResidualConnection(dropout)
    #     # form their sublayers respectively 
    #     self.attentionSublayer = MultiHeadAttentionBlock(d_model, h, dropout)
    #     self.feedForwardSublayer = FeedForwardBlock(d_model, d_ff, dropout)
    
    # def forward(self, x):
    #     for i in range(self.N):
    #         x = self.attentionResidual(x, self.attentionSublayer) # self.attentionSublayer(x, x, x, mask)
    #         x = self.feedForwardResidual(x, self.feedForwardSublayer)
    #     return x 

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residualConnections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask): # avoid interactions with padding words 
        x = self.residualConnections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residualConnections[1](x, self.feed_forward_block)
        return x 
    


class Encoder(nn.Module):

    """
    At the end of the N times for loop, normalize the output
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.encoderBlocks = blocks 
        self.norm = LayerNormalization()
    
    def forward(self, x, src_mask):
        for block in self.encoderBlocks:
            x = block(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.residualConnections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residualConnections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        x = self.residualConnections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, encoder_mask))
        x = self.residualConnections[2](x, self.feed_forward_block)
        return x 


class Decoder(nn.Module):

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.decoderBlocks = blocks 
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for block in self.decoderBlocks:
            x = block(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)


class ProjectLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        [batch, seq_len, d_model] --> [batch, seq_len, vocab_size], with log softmax post-processing
        """
        return torch.log_softmax(self.linear(x), dim=-1)











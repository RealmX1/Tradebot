"""
A from scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences. I tried to make it as clear as
possible to understand and also went through the code
on my youtube channel!
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, feature_num, heads):
        super(SelfAttention, self).__init__()
        self.feature_num = feature_num
        self.heads = heads
        self.head_dim = feature_num // heads

        assert (
            self.head_dim * heads == feature_num
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(feature_num, feature_num)
        self.keys = nn.Linear(feature_num, feature_num)
        self.queries = nn.Linear(feature_num, feature_num)
        self.fc_out = nn.Linear(feature_num, feature_num)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        print(values.shape, keys.shape, query.shape)
        values = self.values(values)  # (N, value_len, feature_num)
        keys = self.keys(keys)  # (N, key_len, feature_num)
        queries = self.queries(query)  # (N, query_len, feature_num)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # set to negative infinity becuase the ensuing softmax will make it 0

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.feature_num ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, feature_num)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, feature_num, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(feature_num, heads)
        self.norm1 = nn.LayerNorm(feature_num)
        self.norm2 = nn.LayerNorm(feature_num)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_num, forward_expansion * feature_num),
            nn.ReLU(),
            nn.Linear(forward_expansion * feature_num, feature_num),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        feature_num,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.feature_num = feature_num
        self.device = device
        self.position_embedding = nn.Embedding(max_length, feature_num)
        '''
            nn.Embedding(num_embeddings, embedding_dim)
            input: (?), IntTensor or LongTensor of arbitrary shape containing the indices to extract
            output: (?, H), H = embedding_dim
        '''

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    feature_num,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (N, seq_length)
        N, seq_length, feature_n = x.shape
        assert feature_n == self.feature_num

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (x + self.position_embedding(positions)) # x + self.position_embedding(positions)
        )
        # out: (N, seq_length, feature_num)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, feature_num, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(feature_num)
        self.attention = SelfAttention(feature_num, heads=heads)
        self.transformer_block = TransformerBlock(
            feature_num, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        feature_num,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        # self.word_embedding = nn.Embedding(trg_vocab_size, feature_num)
        self.position_embedding = nn.Embedding(max_length, feature_num)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(feature_num, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(feature_num, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((x + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        feature_num=22,
        num_layers=4,
        forward_expansion=1, # 4
        heads=2, # 8
        dropout=0,
        device="cuda",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            feature_num,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            feature_num, # need to pad to feature_num
            num_layers,
            heads, # 1?
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # src mask is used to block all positions where the source is padded 
    # -- even after calcualtion in the transformer (at the end of encoder & decoder forward)
    def make_src_mask(self, src):
        # 1 if not padded, 0 if padded
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len, trg_feature_num = trg.shape
        assert trg_feature_num == 1
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        # src_mask = self.make_src_mask(src)
        # (N, 1, 1, src_len)
        trg_mask = self.make_trg_mask(trg)

        padding = torch.zeros((trg.shape[0], trg.shape[1], feature_num-trg.shape[2])).to(self.device)
        padded_trg = torch.cat([trg, padding], dim=2)
        # (N, 1, trg_len, trg_len)
        enc_src = self.encoder(src, None)
        out = self.decoder(padded_trg, enc_src, None, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    N = 100
    seq_len = 32
    feature_num = 22

    # Generate random tensor with values between 0 and 1
    x = torch.rand((N, seq_len, feature_num)).to(device)

    trg = torch.rand(N, seq_len, 1).to(device)
    print('x.shape', x.shape)
    # x.shape (N, src_len)
    # trg.shape (N, trg_len)

    src_pad_idx = -1e20
    trg_pad_idx = -1e20
    src_vocab_size = 10
    trg_vocab_size = 1
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg)
    # out = model(x, trg[:, :-1])
    print(out.shape)
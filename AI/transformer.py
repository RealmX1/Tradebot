"""
Taken from a git-repo
"""

import torch
import torch.nn as nn


# class SelfAttention(nn.Module):
#     def __init__(self, feature_num, heads):
#         super(SelfAttention, self).__init__()
#         self.feature_num = feature_num
#         self.heads = heads
#         self.head_dim = feature_num // heads

#         assert (
#             self.head_dim * heads == feature_num
#         ), "Embedding size needs to be divisible by heads"

#         self.values = nn.Linear(feature_num, feature_num)
#         self.keys = nn.Linear(feature_num, feature_num)
#         self.queries = nn.Linear(feature_num, feature_num)
#         self.fc_out = nn.Linear(feature_num, feature_num)

#     def forward(self, values, keys, query, mask):
#         # Get number of training examples
#         N = query.shape[0]

#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
#         # print(values.shape, keys.shape, query.shape)
#         values = self.values(values)  # (N, value_len, feature_num)
#         keys = self.keys(keys)  # (N, key_len, feature_num)
#         queries = self.queries(query)  # (N, query_len, feature_num)

#         # Split the embedding into self.heads different pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = queries.reshape(N, query_len, self.heads, self.head_dim)

#         # Einsum does matrix mult. for query*keys for each training example
#         # with every other training example, don't be confused by einsum
#         # it's just how I like doing matrix multiplication & bmm

#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#         # queries shape: (N, query_len, heads, heads_dim),
#         # keys shape: (N, key_len, heads, heads_dim)
#         # energy: (N, heads, query_len, key_len)

#         # Mask padded indices so their weights become 0
#         # print('energy.shape: ', energy.shape)
        
#         if mask is not None:
            
#             # print('mask.shape: ', mask.shape)
#             # print(mask[0])
#             energy = energy.masked_fill(mask == 0, float("-1e20")) # set to negative infinity becuase the ensuing softmax will make it 0
        
#         print('energy: ', energy[0])
#         # Normalize energy values similarly to seq2seq + attention
#         # so that they sum to 1. Also divide by scaling factor for
#         # better stability
#         attention = torch.softmax(energy / (self.feature_num ** (1 / 2)), dim=3)
#         # print('attention.shape: ', attention.shape)
#         print('attention: ', attention[0])
#         # attention shape: (N, heads, query_len, key_len)

#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.heads * self.head_dim
#         )
#         print('out: ', out[0])
#         # attention shape: (N, heads, query_len, key_len)
#         # values shape: (N, value_len, heads, heads_dim)
#         # out after matrix multiply: (N, query_len, heads, head_dim), then
#         # we reshape and flatten the last two dimensions.

#         out = self.fc_out(out)
#         # Linear layer doesn't modify the shape, final shape will be
#         # (N, query_len, feature_num)

#         return out
    
class TransformerBlock(nn.Module):
    def __init__(self, feature_num, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(feature_num, heads)
        self.norm1 = nn.LayerNorm(feature_num)
        self.norm2 = nn.LayerNorm(feature_num)

        self.feed_forward = nn.Sequential(
            nn.Linear(feature_num, forward_expansion * feature_num),
            nn.ReLU(),
            nn.Linear(forward_expansion * feature_num, feature_num),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, attn_mask = mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
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
        # x: (N, seq_length, feature_n)
        N, seq_length, feature_n = x.shape
        assert feature_n == self.feature_num

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        positions = self.position_embedding(positions)
        out = self.dropout(
            (x + positions) # x + self.position_embedding(positions)
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
        self.attention = nn.MultiheadAttention(feature_num, heads)
        self.transformer_block = TransformerBlock(
            feature_num, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, attn_mask = trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        input_dim,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        # self.word_embedding = nn.Embedding(output_dim, feature_num)
        self.position_embedding = nn.Embedding(max_length, input_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(input_dim, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # x: (N, seq_length, input_dim) (padded)
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) # /seq_length #?
        x = self.dropout((x + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        src_pad_idx,
        trg_pad_idx,
        feature_num,
        num_layers=4,
        forward_expansion=1, # 4
        heads=2, # 8
        dropout=0,
        device="cuda",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            input_dim,
            feature_num,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            output_dim,
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
        trg_mask = torch.tril(torch.ones((trg_len, trg_len)),diagonal=0).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        # shift_padding = torch.zeros((trg.shape[0], 1, trg.shape[2])).to(self.device)
        # trg = torch.cat([shift_padding, trg], dim=1)
        # there are two options, either padd one empty timestamp, or chagne the mask to start with full zero first row.

        input_dim = src.shape[2]
        # src_mask = self.make_src_mask(src)
        # (N, 1, 1, src_len)
        trg_mask = self.make_trg_mask(trg)
        # print('target mask shape: ', trg_mask.shape)
        # print('target mask: ', trg_mask[0])

        padding = torch.zeros((trg.shape[0], trg.shape[1], input_dim-trg.shape[2])).to(self.device)
        padded_trg = torch.cat([trg, padding], dim=2)
        # (N, 1, trg_len, trg_len)
        enc_src = self.encoder(src, None)
        out = self.decoder(padded_trg, enc_src, None, trg_mask)
        return out
    
    def predict(self, src, trg):
        input_dim = src.shape[2]
        batch_size = trg.shape[0]
        pred_window = trg.shape[1]
        out_dim = trg.shape[2]
        # src_mask = self.make_src_mask(src)
        # (N, 1, 1, src_len)
        trg_mask = self.make_trg_mask(trg)
        # print('target mask shape: ', trg_mask.shape)
        # print('target mask: ', trg_mask[0])

        padding = torch.zeros((batch_size, pred_window, input_dim-out_dim)).to(self.device)
        padded_trg = torch.cat([trg, padding], dim=2)
        # (N, 1, trg_len, trg_len)
        enc_src = self.encoder(src, None)
        out = self.decoder(padded_trg, enc_src, None, trg_mask)

        for i in range (pred_window):
            padded_out = torch.cat([out, padding], dim=2)
            out = self.decoder(padded_out, enc_src, None, trg_mask)
        # print('out shape: ', out.shape)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 100
    seq_len = 32
    feature_num = 16
    num_layers = 2

    # Generate random tensor with values between 0 and 1
    x = torch.rand((N, seq_len, feature_num)).to(device)

    trg = torch.rand(N, seq_len, 1).to(device)
    print('x.shape: ', x.shape)
    # x.shape (N, src_len)
    # trg.shape (N, trg_len)

    src_pad_idx = -1e20
    trg_pad_idx = -1e20
    input_dim = 10
    output_dim = 1
    model = Transformer(
        input_dim,
        output_dim,
        src_pad_idx,
        trg_pad_idx,
        feature_num=feature_num,
        num_layers=num_layers,
        forward_expansion=2, # 4
        heads=4, # 8
        dropout=0,
        device="cuda",
        max_length=100,
    ).to(device)
    out = model(x, trg)
    # out = model(x, trg[:, :-1])
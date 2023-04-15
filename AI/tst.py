import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

'''
c_in:       the number of features (aka variables, dimensions, channels) in the time series dataset. dls.var
c_out:      the number of target classes. dls.c
seq_len:    number of time steps in the time series. dls.len
max_seq_len:useful to control the temporal resolution in long time series to avoid memory issues. Default. None.
d_model:    total dimension of the model (number of features created by the model). Usual values: 128-1024. Default: 128.
n_heads:    parallel attention heads. Usual values: 8-16. Default: 16.
d_k:        size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
d_v:        size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
d_ff:       the dimension of the feedforward network model. Usual values: 256-4096. Default: 256.
dropout:    amount of residual dropout applied in the encoder. Usual values: 0.-0.3. Default: 0.1.
activation: the activation function of intermediate layer, relu or gelu. Default: 'gelu'.
n_layers:   the number of sub-encoder-layers in the encoder. Usual values: 2-8. Default: 3.
fc_dropout: dropout applied to the final fully connected layer. Usual values: 0.-0.8. Default: 0.
y_range:    range of possible y values (used in regression tasks). Default: None
kwargs:     nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
'''

def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Transpose(Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x): 
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    def __repr__(self): 
        if self.contiguous: return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"

# %% ../../nbs/049_models.TST.ipynb 8
class _ScaledDotProductAttention(Module):
    def __init__(self, d_k:int): 
        super().__init__()
        self.d_k = d_k
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                         # scores : [bs x n_heads x q_len x q_len]
        
        # Scale
        scores = scores / (self.d_k ** 0.5)
        
        # Mask (optional)
        if mask is not None: scores.masked_fill_(mask, -1e9)
        
        # SoftMax
        attn = F.softmax(scores, dim=-1)                                    # attn   : [bs x n_heads x q_len x q_len]
        
        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                     # context: [bs x n_heads x q_len x d_v]
        
        return context, attn

# %% ../../nbs/049_models.TST.ipynb 9
class _MultiHeadAttention(Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int):
        super().__init__()
        r"""
        Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor=None):
        
        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        context, attn = _ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)          # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)                                                  # context: [bs x q_len x d_model]
        
        return output, attn

# %% ../../nbs/049_models.TST.ipynb 11
def get_activation_fn(activation):
    if activation == "relu": return nn.ReLU()
    elif activation == "gelu": return nn.GELU()
    else: return activation()
#         raise ValueError(f'{activation} is not available. You can use "relu" or "gelu"')

class _TSTEncoderLayer(Module):
    def __init__(self, q_len:int, d_model:int, n_heads:int, d_k:int=None, d_v:int=None, d_ff:int=256, dropout:float=0.1, 
                 activation:str="gelu"):
        super().__init__()

        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), 
                                get_activation_fn(activation), 
                                nn.Dropout(dropout), 
                                nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))

    def forward(self, src:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_attn(src)      # Norm: batchnorm 

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.batchnorm_ffn(src) # Norm: batchnorm

        return src

# %% ../../nbs/049_models.TST.ipynb 13
class _TSTEncoder(Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, dropout=0.1, activation='gelu', n_layers=1):
        super().__init__()
        
        self.layers = nn.ModuleList([_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, 
                                                            activation=activation) for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output

# %% ../../nbs/049_models.TST.ipynb 14
class TST(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int, max_seq_len:int=None, 
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k:int=None, d_v:int=None,  
                 d_ff:int=256, dropout:float=0.1, act:str="gelu", fc_dropout:float=0., 
                 y_range:tuple=None, verbose:bool=False, p_len = None,device = None, **kwargs):
        super().__init__()
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len = c_out, seq_len
        
        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        # if max_seq_len is not None and seq_len > max_seq_len: # Control temporal resolution
        #     self.new_q_len = True
        #     q_len = max_seq_len
        #     tr_factor = math.ceil(seq_len / q_len)
        #     total_padding = (tr_factor * q_len - seq_len)
        #     padding = (total_padding // 2, total_padding - total_padding // 2)
        #     self.W_P = nn.Sequential(nn.ConstantPad1d(padding, value = 0.), Conv1d(c_in, d_model, kernel_size=tr_factor, padding=0, stride=tr_factor))
        #     pv(f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n', verbose)
        # elif kwargs:
        #     self.new_q_len = True
        #     t = torch.rand(1, 1, seq_len)
        #     q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
        #     self.W_P = nn.Conv1d(c_in, d_model, **kwargs) # Eq 2
        #     pv(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n', verbose)
        # else:
        self.W_P = nn.Linear(c_in, d_model) # Eq 1: projection of feature vectors onto a d-dim vector space

        if p_len is None:
            self.p_len = q_len
        else:
            self.p_len = p_len

        # Positional encoding
        W_pos = torch.empty((q_len, d_model), device=device)
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=act, n_layers=n_layers).to(device)
        self.flatten = nn.Flatten()
        
        # Head
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None, **kwargs):
        # take in z: [bs x d_model x q_len] and output [bs x c_out x p_len]
        layers = [get_activation_fn(act), nn.Flatten()]
        if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(nf, c_out)]
        # if y_range: layers += [SigmoidRange(*y_range)] ??? What is the SigmoidRange here from?
        return nn.Sequential(*layers)    
        

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        if self.new_q_len: u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)                                             # output: [bs x c_out]

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bs = 64
    c_in = 9  # aka channels, features, variables, dimensions
    c_out = 2
    seq_len = 100

    xb = torch.randn(bs, c_in, seq_len).to(device)

    # standardize by channel by_var based on the training set
    xb = (xb - xb.mean((0, 2), keepdim=True)) / xb.std((0, 2), keepdim=True)

    # Settings
    max_seq_len = 100
    d_model = 32
    n_heads = 4
    d_k = d_v = None # if None --> d_model // n_heads
    d_ff = 128
    dropout = 0.1
    activation = "gelu"
    n_layers = 3
    fc_dropout = 0.1
    kwargs = {}



    model = TST(c_in, c_out, seq_len, max_seq_len=max_seq_len, d_model=d_model, n_heads=n_heads,
                d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=activation, n_layers=n_layers,
                fc_dropout=fc_dropout, device=device, **kwargs).to(device)
    out = model(xb)
    print(out.shape)
    print(f'model parameters: {count_parameters(model)}')
     
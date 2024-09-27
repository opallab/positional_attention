import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class MLP(nn.Module): 
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers-1):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim, bias=True))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.mlp(x)
        return out

class MultiheadAttention(nn.Module):
    """
    The following implementation of multihead attention is adapted from
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Multi-Head-Attention
    """
    def __init__(self, embed_dim, num_heads, positional=False, RoPE=False, pos_dim=-1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        assert not (positional and RoPE), "Cannot have both positional encoding and relative positional encoding."
        if positional:
             assert pos_dim >= 1, "Positional dimension must be positive."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.positional = positional
        self.RoPE = RoPE
        self.head_dim = embed_dim // num_heads
        self.d_k = pos_dim if positional else embed_dim

        # Stack all weight matrices 1...h together for efficiency
        self.qk_proj = nn.Linear(self.d_k, 2*self.d_k*num_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qk_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.qk_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)  
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, p=None):
        batch_size, seq_length, _ = x.size()
        qk = self.qk_proj(p) if self.positional else self.qk_proj(x)
        v = self.v_proj(x)

        # Separate Q, K, V from linear output
        if self.positional:
            # With positional attention, Q, K do not depend on the input data,
            # therefore we do not need the batch dim for Q, K
            qk = qk.reshape(seq_length, self.num_heads, 2*self.d_k)
            qk = qk.permute(1, 0, 2) # [Head, SeqLen, EmbedDim]
        else:
            qk = qk.reshape(batch_size, seq_length, self.num_heads, 2*self.d_k)
            qk = qk.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, EmbedDim]
        q, k = qk.chunk(2, dim=-1)
        # If RoPE is true use rotary positional embeddings
        if self.RoPE:
            device = next(self.parameters()).device
            rotary_emb = RotaryPositionalEmbeddings(dim=self.embed_dim).to(device)
            q = rotary_emb(q)
            k = rotary_emb(k)

        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, EmbedDim]

        # Determine value outputs
        values = F.scaled_dot_product_attention(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, EmbedDim]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        out = self.o_proj(values)
        
        return out

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, mlp_num_layers, positional=False, RoPE=False, pos_dim=-1):
        super().__init__()
        
        self.attn = MultiheadAttention(embed_dim, num_heads, positional=positional, RoPE=RoPE, pos_dim=pos_dim)
        self.mlp = MLP(2*embed_dim, mlp_hidden_dim, embed_dim, mlp_num_layers)

    def forward(self, x, p=None):
        out = self.attn(x, p=p)
        out = torch.cat([x, out], dim=-1)
        out = self.mlp(out)
        return out
    

class Transformer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, num_heads, num_layers, mlp_hidden_dim=128, mlp_num_layers=2, positional=False, RoPE=False, pos_dim=-1):
        super().__init__()
        
        self.encoding = nn.Linear(in_dim, embed_dim)
        self.decoding = nn.Linear(embed_dim, out_dim)
        
        transformer_layers = []
        for _ in range(num_layers):
            transformer_layers.append(TransformerLayer(embed_dim=embed_dim,
                                                            num_heads=num_heads,
                                                            mlp_hidden_dim=mlp_hidden_dim,
                                                            mlp_num_layers=mlp_num_layers,
                                                            positional=positional,
                                                            RoPE=RoPE,
                                                            pos_dim=pos_dim))
        self.transformer_layers = nn.ModuleList(transformer_layers)

    def forward(self, x, p=None):
        x = self.encoding(x)
        for layer in self.transformer_layers:
            x = layer(x, p=p)
        out = self.decoding(x)
        return out
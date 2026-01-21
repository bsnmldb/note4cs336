import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        sigma = (2 / (in_features + out_features))**0.5
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3*sigma, b=3*sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in, out in -> ... out")
    
    def accounting_flops(self, shape: torch.Size) -> int:
        in_features = shape[-1]
        assert in_features == self.weight.shape[1], "Input feature size does not match weight shape"
        out_features = self.weight.shape[0]
        batch_size = reduce(shape, '... in -> ...', 'prod').item() // in_features
        return batch_size * in_features * out_features * 2
    
    def accounting_params(self) -> int:
        return self.weight.numel()

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    
    def accounting_params(self) -> int:
        return self.embedding.numel()
    
    def accounting_flops(self, shape: torch.Size) -> int:
        return 0

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.gain
        return x.to(in_dtype)
    
    def accounting_flops(self, shape: torch.Size) -> int:
        d_model = shape[-1]
        batch_size = reduce(shape, '... d_model -> ...', 'prod').item() // d_model
        return batch_size * d_model * 4  # mean, pow, add, mul
    
    def accounting_params(self) -> int:
        return self.gain.numel()
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    @classmethod
    def silu(cls, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))
    
    def accounting_flops(self, shape: torch.Size) -> int:
        d_model = shape[-1]
        assert d_model == self.d_model, "Input feature size does not match d_model"
        batch_size = reduce(shape, '... d_model -> ...', 'prod').item() // d_model
        flops_linear1 = self.linear1.accounting_flops(shape)
        shape_ff = shape[:-1] + (self.d_ff,)
        flops_linear2 = self.linear2.accounting_flops(shape_ff)
        flops_linear3 = self.linear3.accounting_flops(shape)
        flops_silu = batch_size * self.d_ff * 4  # sigmoid, mul
        flops_elementwise = batch_size * self.d_ff  # element-wise multiplication
        return flops_linear1 + flops_linear2 + flops_linear3 + flops_silu + flops_elementwise
    
    def accounting_params(self) -> int:
        return self.linear1.accounting_params() + self.linear2.accounting_params() + self.linear3.accounting_params()
        
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(0, max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i , j -> i j", positions, inv_freq)
        # Cache sin/cos for every position; shape: (max_seq_len, d_k / 2)
        self.register_buffer("sin", torch.sin(sinusoid_inp), persistent=False)
        self.register_buffer("cos", torch.cos(sinusoid_inp), persistent=False)
        
    def forward(self, in_query_or_key: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Pick the rows we need and broadcast across batch/heads dimensions
        sin = self.sin[token_positions].repeat_interleave(2, dim=-1)
        cos = self.cos[token_positions].repeat_interleave(2, dim=-1)
        x1, x2 = in_query_or_key[..., ::2], in_query_or_key[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return in_query_or_key * cos + rotated * sin
    
    def accounting_flops(self, shape: torch.Size) -> int:
        d_k = shape[-1]
        assert d_k == self.d_k, "Input feature size does not match d_k"
        batch_size = reduce(shape, '... d_k -> ...', 'prod').item() // d_k
        return batch_size * d_k * 4  # 2 muls and 2 adds
    
    def accounting_params(self) -> int:
        return 0
    
def stable_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        query: Tensor of shape (batch_size, ..., seq_len, d_k)
        key: Tensor of shape (batch_size, ..., seq_len, d_k)
        value: Tensor of shape (batch_size, ..., seq_len, d_v)
        mask: Optional Tensor of shape (seq_len, seq_len) where positions with False are masked out
    Returns:
        Tensor of shape (batch_size, ..., seq_len, d_v) after applying scaled dot-product attention
    """
    d_k = query.shape[-1]
    norm = torch.rsqrt(torch.tensor(d_k, dtype=query.dtype, device=query.device))
    scores = einsum(query, key, "... i d_k, ... j d_k -> ... i j") * norm
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn_weights = stable_softmax(scores, dim=-1)
    output = einsum(attn_weights, value, "... i j, ... j d_v -> ... i d_v")
    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, positional_encoding=lambda x,y: x):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.positional_encoding = positional_encoding
        
        self.q_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        
    def _slice_multihead(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
    
    def _combine_multihead(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        query = self._slice_multihead(self.q_linear(x))
        query = self.positional_encoding(query, token_positions)
        key = self._slice_multihead(self.k_linear(x))
        key = self.positional_encoding(key, token_positions)
        value = self._slice_multihead(self.v_linear(x))
        
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(query, key, value, mask=causal_mask)
        attn_output_combined = self._combine_multihead(attn_output)
        output = self.out_linear(attn_output_combined)
        return output
    
    def accounting_flops(self, shape: torch.Size) -> int:
        d_model = shape[-1]
        assert d_model == self.d_model, "Input feature size does not match d_model"
        batch_size = reduce(shape, '... d_model -> ...', 'prod').item() // d_model
        flops_q = self.q_linear.accounting_flops(shape)
        flops_k = self.k_linear.accounting_flops(shape)
        flops_v = self.v_linear.accounting_flops(shape)
        shape_multihead = shape[:-1] + (self.num_heads, self.d_k)
        flops_attention = scaled_dot_product_attention(
            torch.empty(shape_multihead),
            torch.empty(shape_multihead),
            torch.empty(shape_multihead)
        ).numel() * 2  # each multiplication and addition counts as a flop
        flops_out = self.out_linear.accounting_flops(shape)
        return flops_q + flops_k + flops_v + flops_attention + flops_out
    
    def accounting_params(self) -> int:
        return (self.q_linear.accounting_params() +
                self.k_linear.accounting_params() +
                self.v_linear.accounting_params() +
                self.out_linear.accounting_params())
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, positional_encoding=lambda x,y: x):
        super().__init__()
        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype,
                                         positional_encoding=positional_encoding)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.rms1(x))
        x = x + self.ffn(self.rms2(x))
        return x
    
    def accounting_flops(self, shape: torch.Size) -> int:
        d_model = shape[-1]
        assert d_model == self.mha.d_model, "Input feature size does not match d_model"
        batch_size = reduce(shape, '... d_model -> ...', 'prod').item() // d_model
        flops_rms1 = self.rms1.accounting_flops(shape)
        flops_mha = self.mha.accounting_flops(shape)
        flops_rms2 = self.rms2.accounting_flops(shape)
        flops_ffn = self.ffn.accounting_flops(shape)
        flops_residuals = batch_size * d_model * 2  # two residual connections
        return flops_rms1 + flops_mha + flops_rms2 + flops_ffn + flops_residuals
    
    def accounting_params(self) -> int:
        return (self.rms1.accounting_params() +
                self.mha.accounting_params() +
                self.rms2.accounting_params() +
                self.ffn.accounting_params())
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.decoders = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype, positional_encoding=self.rope)
            for _ in range(num_layers)
        ])
        self.rms_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for block in self.decoders:
            x = block(x)
        x = self.rms_final(x)
        logits = self.output_linear(x)
        return logits
    
    def accounting_flops(self, shape: torch.Size) -> int:
        seq_len = shape[-2]
        d_model = shape[-1]
        assert d_model == self.decoders[0].mha.d_model, "Input feature size does not match d_model"
        batch_size = reduce(shape, '... d_model -> ...', 'prod').item() // d_model
        flops_embedding = self.embedding.accounting_flops(shape)
        flops_blocks = sum(block.accounting_flops(shape) for block in self.decoders)
        flops_rms_final = self.rms_final.accounting_flops(shape)
        flops_output_linear = self.output_linear.accounting_flops(shape)
        return flops_embedding + flops_blocks + flops_rms_final + flops_output_linear
    
    def accounting_params(self) -> int:
        return (self.embedding.accounting_params() +
                sum(block.accounting_params() for block in self.decoders) +
                self.rms_final.accounting_params() +
                self.output_linear.accounting_params())
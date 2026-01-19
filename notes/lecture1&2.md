# Lecture 1: Intro and Tokenization

## Course Outline
1. Basics (tokenization, architecture, loss function, optimizer)
2. Systems (kernel, parallelism, quantization, activation checkpointing, cpu offloading, inference)
3. Scaling laws (scaling sequence, model complexity, loss metric, parametric form)
4. Data (evaluation, curation, transformation, filtering, deduplication, mixing)
5. Alignment (sft, rl, preference data, synthetic data, verifier)

## Tokenization
1. character-based tokenization
   1. large vocabulary
   2. inefficient use of the vaocabulary (many are quite rare, like 'A' vs an emoji)
2. byte-based tokenization
   1. compression ratio is 1, which means the sequences will be too long
   2. but the context length of a Transformer is limited
3. word-based tokenization
   1. very very large vocabulary, not fixed
   2. some are rare and model won't learn much about them
   3. out-of-vocabulary problem
4. Byte Pair Encoding (BPE)
   1. trade-off between byte-based and word-based tokenization, let the model learn the subword units
   2. start with each byte as a token, and successively merge the most frequent pairs of adjacent tokens

# Lecture 2: PyTorch and FLOPs

## Memory Accounting
1. float32: single precision, 4 bytes
   1. 1 sign, 8 exponent, 23 mantissa
2. float16: half precision, 2 bytes
   1. 1 sign, 5 exponent, 10 mantissa
   2. underflow/overflow issues
3. bfloat16: brain float (by Google Brain, 2019), 2 bytes
   1. 1 sign, 8 exponent, 7 mantissa
   2. same range as float32, but less precision, good for deep learning
4. fp8: variants

```python
float32_info = torch.finfo(torch.float32)
print(float32_info)
float16_info = torch.finfo(torch.float16)
print(float16_info)
bfloat16_info = torch.finfo(torch.bfloat16)
print(bfloat16_info)
```

## Compute Accounting
### Tensor On GPUs
### Tensor Operations
1. tensor storage: sequentially in memory, use metadata to index
2. tensor slicing: **view** on the original tensor, **no data copy**
```python
x = torch.arange(6).view(2,3)
y = x[0]
y = x[:, 1]
y = x.view(3, 2)
y = x.transpose(1, 0)
```
   Note that if the tensor is non-contiguous, some operations may cause error. One can enforce a tensor to be contiguous using `tensor.contiguous()` and copy occurs here.
3. tensor elementwise
4. tensor matmul

### Tensor Einops (a library for tensor operations)
1. einsum
```python
# Notation
x: Float[torch.Tensor, "bnatch seq2 hidden"] = torch.ones(2, 3, 4)

# Old way 
z = x @ y.transpose(-2, -1)
# Use einops
z = torch.einsum("bsh,bth->bst", x, y)
# use dot to broadcast
z = torch.einsum("...sh,...th->...st", x, y)
```
2. reduce
```python
x: Float[torch.Tensor, "bnatch seq2 hidden"] = torch.ones(2, 3, 4)
# Old way
y = x.sum(dim=-1)
# Use einops
y = torch.einsum("...h->...", x)
```
3. rearrange
```python
x: Float[torch.Tensor, "bnatch seq total_hidden"] = torch.ones(2, 3, 8) # actually heads * hidden
x = torch.rearrange(x, "... (h d) -> ... h d", h=2)
# do some operations
x = torch.rearrange(x, "... h d -> ... (h d)")
```

### Operation FLOPs
1. FLOPs: compute operations num
2. FLOP/s: speed --- operation per second, depends on hardware and data type

matrix multiplication FLOPs = 2 * M * N * K
element wise FLOPs = M * N
roughly, FLOPs of a forward pass is 2 * (#tokens) * (#parameters), and it generalizes to Transformers.

Model FLOPs utilization (MFU): (actual FLOP/s) / (promised FLOP/s)
Usually, MFU of >= 0.5 is good, because there is overhead and communication costs.

### Gradients FLOPs
some details ...
summary:
+ Forward pass: 2 (# data points) (# parameters) FLOPs
+ Backward pass: 4 (# data points) (# parameters) FLOPs
+ Total: 6 (# data points) (# parameters) FLOPs

## Model
### Parameter Initialization
```python
w = nn.Parameter(torch.randn(input_dim, hidden_dim) / np.sqrt(input_dim))
```

### Some Practices
randomness
```python
import torch
torch.manual_seed(seed)

import numpy as np
np.random.seed(seed)

import random
random.seed(seed)
```

lazy data loader
```python
data = np.memmap("data.npy", dtype=np.int32)
```

### Optimizer
create an optimizer
Write an optimizer


Memory usage:
```python
num_parameters = (D * D * num_layers) + D
num_activations = B * D * num_layers # hidden states
num_gradients = num_parameters
num_optimizer_states = 2 * num_parameters # Adam
# counting assuming float32
total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)
```

### Train Loop
1. define the model and optimizer
2. get data
3. forward pass
4. loss backward
5. optimizer step
6. optimizer zero grad

### Checkpointing
including saving model weights and optimizer states to resume training later

### Mixed Precision Training
Use float32 by default, but use {bfloat16, fp8} when possible.
Example:
+ {bfloat16, fp8} for forward pass (activations)
+ float32 for the rest (parameters, gradients)

automatic mixed precision (AMP) library...


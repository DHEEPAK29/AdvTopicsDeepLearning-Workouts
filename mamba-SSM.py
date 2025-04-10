# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

# pip install mamba-ssm

# pip install causal-conv1d>=1.4.0

# pip install mamba-ssm[causal-conv1d]

# pip install mamba-ssm[dev]

# git clone https://github.com/state-spaces/mamba.git
# cd mamba
# pip install .

import torch
from mamba_ssm import Mamba

# Define input dimensions
batch_size = 2
sequence_length = 64
model_dim = 16

# Create random input tensor
x = torch.randn(batch_size, sequence_length, model_dim).to("cuda")

# Initialize the Mamba model
model = Mamba(
    d_model=model_dim,  # Model dimension
    d_state=16,         # SSM state dimension
    d_conv=4,           # Local convolution width
    expand=2,           # Block expansion factor
).to("cuda")

# Forward pass
y = model(x)
assert y.shape == x.shape

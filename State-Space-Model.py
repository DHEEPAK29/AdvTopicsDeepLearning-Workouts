# minimal, educational PyTorch example of a discrete-time State-Space Model (SSM) layer

import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_first=True):
        """
        A minimal SSM layer with trainable A, B, C, D matrices.
        
        Args:
            input_dim  (int):  Dimension of each input token u_k
            hidden_dim (int):  Dimension of the hidden state h_k
            output_dim (int):  Dimension of each output token y_k
            batch_first (bool): If True, input shape is (B, T, input_dim)
        """
        super().__init__()
        self.batch_first = batch_first
        
        # SSM parameter matrices
        # A: (hidden_dim x hidden_dim)
        # B: (hidden_dim x input_dim)
        # C: (output_dim x hidden_dim)
        # D: (output_dim x input_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)

    def forward(self, x, h0=None):
        """
        Forward pass through the SSM in time-domain.
        
        Args:
            x (Tensor): Input sequence of shape (B, T, input_dim) if batch_first=True,
                        else (T, B, input_dim).
            h0 (Tensor): Optional initial hidden state of shape (B, hidden_dim).
            
        Returns:
            y (Tensor): Output sequence of shape (B, T, output_dim) if batch_first=True.
            h (Tensor): Final hidden state of shape (B, hidden_dim).
        """
        if not self.batch_first:
            # If the input is (T, B, input_dim), swap to (B, T, input_dim) for convenience
            x = x.transpose(0, 1)  # now x is (B, T, input_dim)

        B, T, _ = x.shape
        hidden_dim = self.A.shape[0]

        # If no initial state is provided, default to zeros
        if h0 is None:
            h0 = torch.zeros(B, hidden_dim, device=x.device, dtype=x.dtype)

        h = h0  # shape (B, hidden_dim)
        outputs = []

        for t in range(T):
            u_t = x[:, t, :]        # (B, input_dim)
            h = h @ self.A.T + u_t @ self.B.T  # h_{k+1} = A h_k + B u_k
            y_t = h @ self.C.T + u_t @ self.D.T  # y_k = C h_k + D u_k
            outputs.append(y_t.unsqueeze(1))    # store this output step

        # Concatenate over time dimension → (B, T, output_dim)
        y = torch.cat(outputs, dim=1)

        return y, h

if __name__ == "__main__":
    # Example usage
    B, T = 2, 5   # batch size and sequence length
    input_dim = 3
    hidden_dim = 4
    output_dim = 3

    ssm = SimpleSSM(input_dim, hidden_dim, output_dim, batch_first=True)
    
    # Fake input: (batch_size=2, seq_len=5, input_dim=3)
    x = torch.randn(B, T, input_dim)
    
    # Forward pass
    y, h_final = ssm(x)  # y -> (B, T, output_dim), h_final -> (B, hidden_dim)

    print("Output (y) shape:", y.shape)
    print("Final hidden state (h_final) shape:", h_final.shape)

"""
Small transformer model for modular arithmetic tasks.
Based on the architecture from Power et al. (2022) and Chan et al. (2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModularArithmeticTransformer(nn.Module):
    """
    1-layer transformer for modular arithmetic (a + b) mod p.
    
    Architecture:
    - Token embedding: map each integer to a d_model-dimensional vector
    - Positional encoding: learned or fixed
    - 1 transformer encoder layer with multi-head attention
    - Output head: project to p classes
    
    This is intentionally small to enable grokking observation.
    """
    
    def __init__(
        self,
        prime: int = 59,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.prime = prime
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Token embeddings (0 to p-1)
        self.token_embed = nn.Embedding(prime, d_model)
        
        # Positional embeddings (2 positions: a and b)
        self.pos_embed = nn.Embedding(2, d_model)
        
        # Transformer layer components (explicit for attention extraction)
        # Using a single layer as specified (n_layers=1)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
        # Output head: use the sum of both position representations
        self.output_head = nn.Linear(d_model, prime)
        
        # Final Layer norm (optional, but keeping it to match previous architecture if it was applied at the end)
        self.ln = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 2) with values in [0, prime)
            return_attn: If True, returns a tuple of (logits, attention_weights)
        
        Returns:
            Logits of shape (batch, prime) if return_attn is False,
            otherwise (logits, attn_weights)
        """
        batch_size = x.shape[0]
        
        # Token embeddings
        tok = self.token_embed(x)  # (batch, 2, d_model)
        
        # Positional embeddings
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.pos_embed(positions)  # (batch, 2, d_model)
        
        # Combine
        h = tok + pos  # (batch, 2, d_model)
        
        # Transformer layer
        # Pre-LN or Post-LN depending on previous PyTorch default (default is Post-LN in older versions, but let's do standard Pre-LN/Post-LN)
        # PyTorch TransformerEncoderLayer default is Post-LN: x = x + attn(x), x = norm(x)
        # Let's match the standard post-norm of PyTorch TransformerEncoderLayer(batch_first=True, norm_first=False)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True)
        h = self.ln1(h + attn_out)

        mlp_out = self.mlp(h)
        h = self.ln2(h + mlp_out)

        # Final layer norm
        h = self.ln(h)
        
        # Pool across positions (mean) and predict
        h_pooled = h.mean(dim=1)  # (batch, d_model)
        logits = self.output_head(h_pooled)  # (batch, prime)
        
        if return_attn:
            return logits, attn_weights

        return logits
    
    def get_weight_norm(self) -> float:
        """
        Get total L2 norm of all parameters.
        This metric often spikes before generalization, then decreases when the model
        cleans up representations (the grokking phase).

        Returns:
            The square root of the sum of squared parameter values.
        """
        return sum(p.norm().item() ** 2 for p in self.parameters()) ** 0.5
    
    def get_embedding_fourier_spectrum(self) -> torch.Tensor:
        """
        Compute the Fourier spectrum of the token embedding matrix.
        This allows tracking whether the embeddings learn the circle-representation
        required to perform modular arithmetic via trigonometric identities.

        Returns:
            A tensor of shape (prime, d_model) containing the magnitude of the DFT
            of each embedding dimension.
        """
        W = self.token_embed.weight.detach()  # (prime, d_model)
        # DFT along the token dimension
        spectrum = torch.fft.fft(W, dim=0).abs()
        return spectrum
    
    def get_embedding_rank(self) -> float:
        """Compute effective rank of the embedding matrix."""
        W = self.token_embed.weight.detach()
        s = torch.linalg.svdvals(W)
        s = s / s.sum()
        s = s[s > 1e-10]
        entropy = -(s * torch.log(s)).sum()
        return torch.exp(entropy).item()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ModularArithmeticTransformer()
    print(f"Model parameters: {count_parameters(model):,}")
    
    x = torch.randint(0, 59, (4, 2))
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"Weight norm: {model.get_weight_norm():.2f}")
    print(f"Embedding rank: {model.get_embedding_rank():.2f}")

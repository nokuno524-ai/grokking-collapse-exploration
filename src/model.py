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
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head: use the sum of both position representations
        self.output_head = nn.Linear(d_model, prime)
        
        # Layer norm
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 2) with values in [0, prime)
        
        Returns:
            Logits of shape (batch, prime)
        """
        batch_size = x.shape[0]
        
        # Token embeddings
        tok = self.token_embed(x)  # (batch, 2, d_model)
        
        # Positional embeddings
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.pos_embed(positions)  # (batch, 2, d_model)
        
        # Combine
        h = tok + pos  # (batch, 2, d_model)
        
        # Transformer
        h = self.transformer(h)  # (batch, 2, d_model)
        h = self.ln(h)
        
        # Pool across positions (mean) and predict
        h = h.mean(dim=1)  # (batch, d_model)
        logits = self.output_head(h)  # (batch, prime)
        
        return logits
    
    def get_weight_norm(self) -> float:
        """Get total L2 norm of all parameters."""
        return sum(p.norm().item() ** 2 for p in self.parameters()) ** 0.5
    
    def get_embedding_fourier_spectrum(self) -> torch.Tensor:
        """
        Compute the Fourier spectrum of the token embedding matrix.
        Returns the magnitude of the DFT of each embedding dimension.
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
        entropy = -(s * torch.log(s + 1e-10)).sum()
        return torch.exp(entropy).item()

    def get_attention_snapshots(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for a batch of inputs from the first transformer layer.
        Returns:
            Attention weights of shape (batch, n_heads, seq_len, seq_len)
        """
        batch_size = x.shape[0]
        tok = self.token_embed(x)
        positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos = self.pos_embed(positions)
        h = tok + pos

        # We extract from the first encoder layer
        # The self_attn returns (attn_output, attn_weights) when need_weights=True
        # attn_weights shape: (batch, seq_len, seq_len) if average_attn_weights=True (default in some versions)
        # We need per-head weights, so we set average_attn_weights=False (if supported) or rely on the extraction
        layer = self.transformer.layers[0]

        # manual extraction using self_attn
        attn_output, attn_weights = layer.self_attn(
            h, h, h,
            need_weights=True,
            average_attn_weights=False
        )
        # For batch_first=True, PyTorch MultiheadAttention returns (batch, num_heads, seq_len, seq_len)
        # when average_attn_weights=False.
        return attn_weights.detach()

    def get_svd_spectra(self) -> dict:
        """
        Get SVD spectra (singular values) for major weight matrices.
        Returns:
            Dictionary mapping layer names to their singular values (list of floats).
        """
        spectra = {}
        with torch.no_grad():
            W_embed = self.token_embed.weight.detach()
            spectra["token_embed"] = torch.linalg.svdvals(W_embed).cpu().tolist()

            W_out = self.output_head.weight.detach()
            spectra["output_head"] = torch.linalg.svdvals(W_out).cpu().tolist()

            # Extract from first transformer layer
            layer = self.transformer.layers[0]

            # W_q, W_k, W_v are combined in in_proj_weight for MultiheadAttention
            if layer.self_attn.in_proj_weight is not None:
                W_in_proj = layer.self_attn.in_proj_weight.detach()
                spectra["attn_in_proj"] = torch.linalg.svdvals(W_in_proj).cpu().tolist()

            if layer.self_attn.out_proj.weight is not None:
                W_out_proj = layer.self_attn.out_proj.weight.detach()
                spectra["attn_out_proj"] = torch.linalg.svdvals(W_out_proj).cpu().tolist()

            spectra["ff_1"] = torch.linalg.svdvals(layer.linear1.weight.detach()).cpu().tolist()
            spectra["ff_2"] = torch.linalg.svdvals(layer.linear2.weight.detach()).cpu().tolist()

        return spectra


GrokkingTransformer = ModularArithmeticTransformer


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

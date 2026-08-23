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
        """
        Initialize the ModularArithmeticTransformer.

        Args:
            prime: The modulo/vocabulary size.
            d_model: Dimensionality of embeddings and hidden states.
            n_heads: Number of attention heads.
            d_ff: Dimensionality of feedforward network.
            n_layers: Number of transformer blocks.
            dropout: Dropout probability.
        """
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

    def get_attention_entropy(self) -> float:
        """
        Compute mean entropy of attention weights across all heads in the first layer,
        evaluated over the vocabulary. Serves as an early-warning metric for structure formation.
        """
        # PyTorch TransformerEncoderLayer uses MultiheadAttention under the hood.
        # W_q, W_k are slices of in_proj_weight
        attn = self.transformer.layers[0].self_attn
        if attn.in_proj_weight is None:
            return 0.0 # fallback for non-standard init

        W_q = attn.in_proj_weight[:self.d_model, :].detach()
        W_k = attn.in_proj_weight[self.d_model:2*self.d_model, :].detach()

        # Estimate attention using W_q W_k^T on embeddings
        E = self.token_embed.weight.detach() # (prime, d_model)

        Q = E @ W_q.T # (prime, d_model)
        K = E @ W_k.T # (prime, d_model)

        # Reshape to heads
        head_dim = self.d_model // self.n_heads
        Q = Q.view(self.prime, self.n_heads, head_dim)
        K = K.view(self.prime, self.n_heads, head_dim)

        # Attention scores per head: (prime, prime, n_heads)
        scores = torch.einsum('ihd,jhd->ijh', Q, K) / math.sqrt(head_dim)

        probs = F.softmax(scores, dim=1) # softmax over keys j

        # Entropy: -sum p log p
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1) # (prime, n_heads)

        # Mean entropy over tokens and heads
        return entropy.mean().item()

    def get_hidden_activation_rank(self) -> float:
        """
        Compute effective rank of hidden activations.
        We approximate this by looking at the value matrix projection of embeddings
        as a proxy for hidden state capacity, to avoid needing a forward pass hook.
        """
        attn = self.transformer.layers[0].self_attn
        if attn.in_proj_weight is None:
            return 0.0

        W_v = attn.in_proj_weight[2*self.d_model:, :].detach()
        E = self.token_embed.weight.detach()

        V = E @ W_v.T

        s = torch.linalg.svdvals(V)
        s = s / (s.sum() + 1e-10)
        entropy = -(s * torch.log(s + 1e-10)).sum()
        return torch.exp(entropy).item()


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

import torch
import torch.nn as nn
import torch.nn.functional as F

class JLensAnalyzer:
    """
    J-Lens Analyzer to extract intermediate representations and project them
    to the output vocabulary space to measure verbalizable dimensionality and coherence.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def extract_representations(self, x):
        """
        Extract explicitly available intermediate representations:
        1. Embeddings (tok + pos)
        2. Transformer output (h)
        3. Layer Norm output (ln(h))
        """
        batch_size = x.shape[0]

        with torch.no_grad():
            # 1. Embeddings
            tok = self.model.token_embed(x)
            positions = torch.arange(2, device=x.device).unsqueeze(0).expand(batch_size, -1)
            pos = self.model.pos_embed(positions)
            emb = tok + pos

            # 2. Transformer output
            transformer_out = self.model.transformer(emb)

            # 3. Layer norm output
            ln_out = self.model.ln(transformer_out)

        return {
            'embedding': emb,
            'transformer': transformer_out,
            'layer_norm': ln_out
        }

    def project_to_vocabulary(self, representations):
        """
        Project intermediate representations to vocabulary space using the model's output head.
        """
        projections = {}
        with torch.no_grad():
            for name, rep in representations.items():
                # Mean pool across positions
                pooled_rep = rep.mean(dim=1)
                logits = self.model.output_head(pooled_rep)
                projections[name] = logits
        return projections

    def compute_metrics(self, projections, x):
        """
        Compute J-space metrics like entropy and rank for projected outputs.
        """
        metrics = {}
        for name, logits in projections.items():
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

            # Compute effective rank based on SVD of the representations in vocabulary space
            s = torch.linalg.svdvals(logits - logits.mean(dim=0))
            s = s / (s.sum() + 1e-10)
            svd_entropy = -(s * torch.log(s + 1e-10)).sum()
            rank = torch.exp(svd_entropy).item()

            metrics[name] = {
                'entropy': entropy,
                'rank': rank
            }
        return metrics

    def analyze(self, x):
        """
        Run the complete J-Lens analysis on input x.
        """
        reps = self.extract_representations(x)
        projs = self.project_to_vocabulary(reps)
        metrics = self.compute_metrics(projs, x)
        return metrics

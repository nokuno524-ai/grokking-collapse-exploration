import torch
from src.model import ModularArithmeticTransformer
from src.analysis.attention import AttentionExtractor, compute_attention_entropy, compute_head_specialization_clustering

def main():
    model = ModularArithmeticTransformer(n_layers=2)
    x = torch.randint(0, 59, (4, 2))

    with AttentionExtractor(model) as extractor:
        out = model(x)

    print(f"Extracted layers: {list(extractor.maps.keys())}")

    w0 = extractor.maps[0]
    print(f"Layer 0 map shape: {w0.shape}")

    entropy = compute_attention_entropy(w0)
    print(f"Entropy shape: {entropy.shape}")

    clusters = compute_head_specialization_clustering([extractor.maps[0], extractor.maps[1]])
    print(f"Clusters: {clusters}")

if __name__ == "__main__":
    main()

with open("src/transplant_rescue.py", "r") as f:
    content = f.read()
content = content.replace("pat = f\"^transformer\\.layers\\.{layer_idx}\\.self_attn\\.(in_proj_(weight|bias)|out_proj\\.weight)$\"", "pat = f\"^transformer\\\\.layers\\\\.{layer_idx}\\\\.self_attn\\\\.(in_proj_(weight|bias)|out_proj\\\\.weight)$\"")
with open("src/transplant_rescue.py", "w") as f:
    f.write(content)

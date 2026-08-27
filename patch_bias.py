with open("src/transplant_rescue.py", "r") as f:
    content = f.read()

# Fix the bug with in_proj_bias shape determination
# Instead of d_model = base_sd[k].shape[-1], we can infer it correctly based on the shape length.
bug_search = """                d_model = base_sd[k].shape[-1]
                head_dim = d_model // n_heads"""

bug_replace = """                if len(base_sd[k].shape) == 1:
                    # bias is shape (3 * d_model)
                    d_model = base_sd[k].shape[0] // 3
                else:
                    d_model = base_sd[k].shape[-1]
                head_dim = d_model // n_heads"""

content = content.replace(bug_search, bug_replace)

with open("src/transplant_rescue.py", "w") as f:
    f.write(content)

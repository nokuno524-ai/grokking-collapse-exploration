import hashlib

with open("src/data.py", "r") as f:
    content = f.read()

content = content.replace(
"""        gen_seed = hash((seed, "collapse_generation", depth)) % (2**32)""",
"""        seed_str = f"{seed}_collapse_generation_{depth}"
        gen_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)""")

content = content.replace("import torch", "import torch\nimport hashlib")

with open("src/data.py", "w") as f:
    f.write(content)

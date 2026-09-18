import re

with open("src/train.py", "r") as f:
    content = f.read()

# Remove the unused get_batch function definition to clean up
content = re.sub(
    r"    def get_batch\(step, iter_A, iter_B\):.*?(?=\n    # Create model)",
    "",
    content,
    flags=re.DOTALL
)

with open("src/train.py", "w") as f:
    f.write(content)

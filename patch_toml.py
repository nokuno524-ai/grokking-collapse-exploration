with open("pyproject.toml", "r") as f:
    content = f.read()

content = content.replace(
    '"matplotlib>=3.7",\n]',
    '"matplotlib>=3.7",\n    "pandas>=2.0",\n    "scipy>=1.10",\n    "tabulate>=0.9",\n]'
)

with open("pyproject.toml", "w") as f:
    f.write(content)

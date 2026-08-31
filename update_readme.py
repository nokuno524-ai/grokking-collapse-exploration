import re

with open("README.md", "r") as f:
    readme = f.read()

new_section = """
## Analysis & Grokkit CLI

This repository contains a dedicated package, `grokkit`, for reproducibly converting raw experiment logs into standard reports, tables, and figures.

You can run the CLI via:

```bash
grokkit analyze <run_dir>
grokkit compare <dirs...>
grokkit cliff <run_dir>
```

For detailed usage, see [docs/USAGE.md](docs/USAGE.md).
"""

if "## Analysis & Grokkit CLI" not in readme:
    readme += "\n" + new_section
    with open("README.md", "w") as f:
        f.write(readme)

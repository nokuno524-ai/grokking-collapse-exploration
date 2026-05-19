import glob
import os

files = glob.glob("src/**/*.py", recursive=True) + glob.glob("analysis/**/*.py", recursive=True) + glob.glob("tools/**/*.py", recursive=True)

for file in files:
    with open(file, "r") as f:
        lines = f.readlines()

    has_use = any("matplotlib.use" in line for line in lines)
    has_plt = any("import matplotlib.pyplot" in line for line in lines)

    if has_use and has_plt:
        # We need to make sure use() comes before pyplot
        use_idx = next(i for i, line in enumerate(lines) if "matplotlib.use" in line)
        plt_idx = next(i for i, line in enumerate(lines) if "import matplotlib.pyplot" in line)

        if plt_idx < use_idx:
            # Swap them basically, or just move plt below use
            plt_line = lines.pop(plt_idx)
            # Recompute use_idx because we popped
            use_idx = next(i for i, line in enumerate(lines) if "matplotlib.use" in line)
            lines.insert(use_idx + 1, plt_line)

            with open(file, "w") as f:
                f.writelines(lines)
            print(f"Fixed matplotlib order in {file}")

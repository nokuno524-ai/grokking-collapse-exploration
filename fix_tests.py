import os
import glob

for test_file in glob.glob("tests/*.py"):
    with open(test_file, "r") as f:
        content = f.read()
    content = content.replace("from src.grokkit", "from grokkit")
    with open(test_file, "w") as f:
        f.write(content)

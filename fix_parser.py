with open("src/grokkit/parser.py", "r") as f:
    text = f.read()

# Make sure config is truly flattened
text = text.replace('if "config" in data:',
'''if "config" in data:
                # also ensure keys are top level
                for k, v in data["config"].items():
                    data[k] = v''')

with open("src/grokkit/parser.py", "w") as f:
    f.write(text)

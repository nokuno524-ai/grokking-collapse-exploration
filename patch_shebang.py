with open("scripts/visualize_attention.py", "r") as f:
    code = f.read()

code = code.replace("#!/usr/8in/env python3", "#!/usr/bin/env python3")

with open("scripts/visualize_attention.py", "w") as f:
    f.write(code)

with open("src/cli_data_degradation.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print("\\n\nCorrelation Table:")' in line or 'print("\\n\n' in line:
        pass # skip broken lines
    elif 'Correlation Table:")' in line and 'print' not in line:
        new_lines.append('        print("\\nCorrelation Table:")\n')
    else:
        new_lines.append(line)

with open("src/cli_data_degradation.py", "w") as f:
    f.writelines(new_lines)

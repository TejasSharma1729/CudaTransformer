import os
import re
import glob

def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Very simple heuristic for C++ method checking
        # Looking for return_type method_name(...)
        # excluding common keywords
        m = re.match(r'^ *(?:virtual |inline |static )?(?:[A-Za-z_][A-Za-z0-9_<>,: *&]+ )+([A-Za-z_][A-Za-z0-9_]*|operator\(\)|operator[A-Za-z_]+)\([^)]*\)(?: *const| *override| *final)? *(?:{|;).*$', line)
        if m and " else " not in line and " if " not in line and "for " not in line and "while " not in line and "return " not in line and "#define" not in line and "typedef" not in line:
            # Check if previous line had a docstring or comment
            has_doc = i > 0 and ('*/' in lines[i-1] or '///' in lines[i-1] or '//' in lines[i-1])
            # Check if it's inside a function block
            if not has_doc:
                indent = re.match(r'^( *)', line).group(1)
                func_name = m.group(1)
                out_lines.append(f"{indent}/**\n")
                out_lines.append(f"{indent} * @brief Execute {func_name} operation.\n")
                out_lines.append(f"{indent} */\n")
        out_lines.append(line)
        i += 1
        
    with open(filepath, 'w') as f:
        f.writelines(out_lines)

for f in glob.glob("cuda_transformer/*.cu"):
    process_file(f)


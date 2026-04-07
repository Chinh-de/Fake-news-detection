import os, glob, re

target_dir = r'd:/Study_space/Ki8/PBL7/MRCD/src'

for filepath in glob.glob(os.path.join(target_dir, '**', '*.py'), recursive=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if ' ' in content:
        print(f'Found in {filepath}')
        lines = content.split('\n')
        new_lines = []
        skip_mode = False
        empty_lines = 0
        for line in lines:
            if ' ' in line:
                skip_mode = True
                continue
            
            if skip_mode:
                if re.match(r'^\s*$', line):
                    # End skip mode on empty line after Flow
                    skip_mode = False
                    continue
                elif re.match(r'^\s*(Args|Returns|Yields|Raises):', line) or line.strip() == '"""':
                    skip_mode = False
                elif re.match(r'^\s*\d+\.', line) or re.match(r'^\s*-', line):
                    continue
                elif re.match(r'^\s*[a-zA-Z]', line):
                    # Sometimes flow steps wrap, let's just skip anything that looks like text during skip mode.
                    continue
                else:
                    skip_mode = False
            
            if not skip_mode:
                new_lines.append(line)
        
        cleaned_content = '\n'.join(new_lines)
        
        # Fixing any messy quotes
        cleaned_content = re.sub(r'\n\s*\n\s*"""', r'\n    """', cleaned_content)
        cleaned_content = re.sub(r'"""\n    """', r'"""', cleaned_content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print(f'Cleaned {filepath}')

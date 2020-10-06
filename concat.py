import os
from pathlib import Path
with open('output_file.txt','w+') as file:
    for p in  Path('classes-dex2jar').glob('./**/*.java'):
        if p.is_file():
            a = str(open(p, 'r').read())
            file.write(a)

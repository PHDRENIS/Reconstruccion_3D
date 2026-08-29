#!/usr/bin/env python3
"""Remove remaining irrelevant rows from Table 2."""

import os

base = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'IPN', 'TT')
for d in os.listdir(base):
    if 'Reporte' in d and 'final' in d.lower():
        tex_file = os.path.join(base, d, 'main.tex')
        break

tex = open(tex_file, encoding='utf-8').read()

# 2) Remove 'Simple Buildings' row
sb = tex.find('3D Reconstruction of Simple Buildings')
if sb > 0:
    limit = tex.find('No incluye techos planos', sb)
    end = tex.find(r'\\', limit + 5) + 2
    tex = tex[:sb] + tex[end:]
    print('2. Simple Buildings removed')

# 3) Remove 'Displacement Measurement' row  
dm = tex.find('Displacement Measurement and 3D Reconstruction of Segmental')
if dm > 0:
    limit2 = tex.find('No eval', dm)
    end2 = tex.find(r'\\', limit2 + 5) + 2
    tex = tex[:dm] + tex[end2:]
    print('3. Retaining Wall removed')
else:
    print('3. NOT FOUND - trying alt search')
    # Try searching more broadly
    for phrase in ['Displacement', 'Retaining', 'Segmental']:
        idx = tex.find(phrase)
        if idx > 0:
            print(f'   Found "{phrase}" at {idx}')
            print(f'   Context: {tex[idx:idx+100]}')

open(tex_file, 'w', encoding='utf-8').write(tex)
print('Done')

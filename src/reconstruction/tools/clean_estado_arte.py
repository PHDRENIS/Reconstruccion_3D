#!/usr/bin/env python3
"""Remove irrelevant papers from estado del arte tables."""

import os, re

base = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'IPN', 'TT')
for d in os.listdir(base):
    if 'Reporte' in d and 'final' in d.lower():
        tex_file = os.path.join(base, d, 'main.tex')
        break

tex = open(tex_file, encoding='utf-8').read()
changes = 0

# === 1) Remove Table 1 entirely (all 3 CAD/hologram papers are irrelevant) ===
# Find the longtblr blocks
tbl1_start = tex.find(r'\begin{longtblr}[', tex.find('% ----- TABLA 1'))
tbl1_end = tex.find(r'\end{longtblr}', tbl1_start) + len(r'\end{longtblr}')

if tbl1_start > 0 and 'eCAD-Net' in tex[tbl1_start:tbl1_end]:
    # Include the comment line before it
    comment_line = tex.rfind('% ----- TABLA 1', 0, tbl1_start)
    tex = tex[:comment_line] + tex[tbl1_end+2:]
    changes += 1
    print(f'1. Table 1 removed (chars {comment_line}-{tbl1_end})')

# === 2) Remove '3D Reconstruction of Simple Buildings' row from Table 2 ===
# Find its row in the table
tbl2_start = tex.find(r'\begin{longtblr}[', tex.find('Parte 2: Reconstrucc'))
tbl2_section = tex[tbl2_start:tbl2_start + 5000]

# Find the row with 'Simple Buildings'
sb_start = tex.find('3D Reconstruction of Simple Buildings', tbl2_start)
if sb_start > 0:
    # Find the \\ at the end of this row
    # Go forward to find the \\ that ends the row (after the Limitaciones column)
    row_end = tex.find(r'\\', sb_start + 100)
    # Need to find the actual end of the row - it's a multi-line row
    # Find 'Limitado a edificios' which is the last column
    limit_line = tex.find('Limitado a edificios', sb_start)
    row_end = tex.find(r'\\', limit_line + 10) + 2
    tex = tex[:sb_start] + tex[row_end:]
    changes += 1
    print(f'2. Simple Buildings removed')

# === 3) Remove 'Displacement Measurement' row from Table 2 ===
dm_start = tex.find('Displacement Measurement and 3D Reconstruction', tbl2_start)
if dm_start > 0:
    limit_line = tex.find('No eval', dm_start)
    row_end = tex.find(r'\\', limit_line + 10) + 2
    tex = tex[:dm_start] + tex[row_end:]
    changes += 1
    print(f'3. Retaining Wall removed')

# === 4) Remove line 527 text that references "Tablas 1 y 2" ===
old_527 = 'El análisis de los trabajos presentados en las Tablas 1 y 2 revela una clara tendencia'
new_527 = 'El análisis de los trabajos presentados en las Tablas del estado del arte revela una clara tendencia'
if old_527 in tex:
    tex = tex.replace(old_527, new_527)
    changes += 1
    print('4. Updated text reference (Tablas 1 y 2 -> Tablas)')

open(tex_file, 'w', encoding='utf-8').write(tex)
print(f'Done: {changes} changes made')

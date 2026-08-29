#!/usr/bin/env python3
"""Add Yosinski citation justification for fine-tuning hyperparameters."""
import os

base = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'IPN', 'TT')
for d in os.listdir(base):
    if 'Reporte' in d and 'final' in d.lower():
        tex_file = os.path.join(base, d, 'main.tex')
        break

tex = open(tex_file, encoding='utf-8').read()

# Find the exact text to replace (L690)
old = 'configuraci\u00f3n de aumentaci\u00f3n de datos est\u00e1ndar de Ultralytics:'

if old in tex:
    new = old + '\n\nLa selecci\u00f3n de hiperpar\u00e1metros para este fine-tuning se fundament\u00f3 en el principio de \\textbf{transferibilidad de caracter\u00edsticas} establecido por \\cite{yosinski2014transferable}: las capas iniciales de una red convolucional aprenden detectores de caracter\u00edsticas generales (bordes, esquinas, texturas) que son transferibles entre dominios, mientras que las capas superiores se especializan en el dominio de entrenamiento original. Con base en este principio, se utiliz\u00f3 una \\textbf{tasa de aprendizaje 20 veces menor} a la del entrenamiento original (lr0=0.0005 vs 0.01) para preservar los pesos del backbone pre-entrenado sin destruir el conocimiento adquirido en SUN RGB-D. Asimismo, se reemplaz\u00f3 el optimizador SGD por \\textbf{AdamW}, cuyo mecanismo de tasas adaptativas por par\u00e1metro permite una convergencia m\u00e1s r\u00e1pida y estable en escenarios de fine-tuning con conjuntos de datos limitados (319 im\u00e1genes).'
    tex = tex.replace(old, new)
    open(tex_file, 'w', encoding='utf-8').write(tex)
    print('OK: Fine-tuning justification added')
else:
    print('NOT FOUND')
    # Debug: find close matches
    if 'Ultralytics:' in tex:
        idx = tex.find('Ultralytics:')
        print('Found Ultralytics at', idx)
        snippet = tex[max(0,idx-80):idx+30]
        print('Context:', repr(snippet[:120]))

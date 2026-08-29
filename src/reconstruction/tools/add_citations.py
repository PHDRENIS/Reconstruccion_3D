#!/usr/bin/env python3
"""Add 5 new citations to LaTeX methodology sections."""
import os, re

base = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'IPN', 'TT')
for d in os.listdir(base):
    if 'Reporte' in d and 'final' in d.lower():
        tex_file = os.path.join(base, d, 'main.tex')
        break

tex = open(tex_file, encoding='utf-8').read()
changes = 0

# 1) ICP Point-to-Plane - Marco Teorico
old1 = 'alineaci\u00f3n m\u00e1s precisa entre fotogramas sucesivos.'
if old1 in tex:
    repl1 = old1 + ' Esta variante, propuesta por \\cite{rusinkiewicz2001efficient}, minimiza la distancia punto-a-plano en lugar de punto-a-punto, ofreciendo una convergencia m\u00e1s r\u00e1pida en superficies planas como paredes y mobiliario.'
    tex = tex.replace(old1, repl1)
    changes += 1
    print('1. ICP citation added')

# 2) TSDF - Marco Teorico, find the sentence that ends with "esculpir una superficie"
for phrase in ['una superficie s\u00f3lida.', 'superficie s\u00f3lida.']:
    if phrase in tex:
        repl2 = phrase[:-1] + ', es el est\u00e1ndar de facto para la fusi\u00f3n de mapas de profundidad en tiempo real \\cite{curless1996volumetric}.'
        tex = tex.replace(phrase, repl2)
        changes += 1
        print('2. TSDF citation added')
        break

# 3) FastSAM - Metodologia, find FastSAM mention
if 'FastSAM' in tex:
    # Find the end of the sentence that introduces FastSAM
    for ending in ['proceso tedioso y propenso a errores.', 'proceso tedioso y propenso a errores']:
        if ending in tex:
            repl3 = ending[:-1] + ', ya que el modelo FastSAM \\cite{zhao2023fastsam} \u2014una variante optimizada del modelo Segment Anything (SAM) que prioriza la velocidad de inferencia\u2014 permite procesar las 319 im\u00e1genes en menos de dos minutos sin requerir anotaci\u00f3n manual.'
            tex = tex.replace(ending, repl3)
            changes += 1
            print('3. FastSAM citation added')
            break
    else:
        print('3. FastSAM: trying alt location...')
        # Just append citation to first FastSAM mention
        idx = tex.find('FastSAM')
        if idx > 0:
            # Find the end of that paragraph/sentence
            next_period = tex.find('.', idx + 50)
            if next_period > 0:
                before = tex[:next_period]
                after = tex[next_period:]
                tex = before + ' \\cite{zhao2023fastsam}' + after
                changes += 1
                print('3. FastSAM citation appended at end of sentence')

# 4) Taubin - Post-processing
for phrase in ['preservando las dimensiones m\u00e9tricas de la escena.', 'dimensiones m\u00e9tricas de la escena.']:
    if phrase in tex:
        repl4 = phrase[:-1] + ' \\cite{taubin1995signal}.'
        tex = tex.replace(phrase, repl4)
        changes += 1
        print('4. Taubin citation added')
        break
else:
    print('4. Taubin: not found')

# 5) Quadric decimation - Post-processing
for phrase in ['algoritmo de decimaci\u00f3n cuadr\u00e1tica', 'decimaci\u00f3n cuadr\u00e1tica']:
    if phrase in tex:
        repl5 = phrase + ' (Quadric Error Metrics) \\cite{garland1997surface}'
        tex = tex.replace(phrase, repl5)
        changes += 1
        print('5. Quadric decimation citation added')
        break
else:
    print('5. Quadric: not found')

open(tex_file, 'w', encoding='utf-8').write(tex)
print(f'Done: {changes}/5 citations added')

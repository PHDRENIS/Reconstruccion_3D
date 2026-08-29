#!/usr/bin/env python3
"""
Analisis de Consistencia Geometrica Global - Version 2 (nube original)
Detecta planos grandes con RANSAC y calcula angulos/distancias entre paredes.
Usa la nube de puntos original (no la malla Poisson) para mayor fidelidad.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import math

INPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Posible reconstruccion final")
OUTPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Metricas Reconstruccion")

INPUT_FILES = [
    ("reconstruction_cloud_cleaned.ply", "Nube Limpia Original (reconstruction_cloud_cleaned)"),
    ("reconstruction_cloud.ply", "Nube Original (reconstruction_cloud)"),
]

def analyze_large_planes(pcd_path, label):
    print(f"\n{'='*70}")
    print(f"ANALIZANDO: {label}")
    print(f"{'='*70}")
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        print(f"ERROR: No se pudo cargar {pcd_path}")
        return None
    
    print(f"Puntos originales: {len(pcd.points):,}")
    
    # Voxel downsample moderado
    pcd_ds = pcd.voxel_down_sample(0.02)
    pts = np.asarray(pcd_ds.points)
    print(f"Puntos despues de voxel (0.02m): {len(pts):,}")
    
    if len(pts) < 1000:
        print("ERROR: Puntos insuficientes")
        return None
    
    # RANSAC iterativo con criterios de area
    remaining = pcd_ds
    all_planes = []
    MAX_PLANES = 12
    
    for i in range(MAX_PLANES):
        if len(remaining.points) < 1000:
            break
        
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.08, ransac_n=3, num_iterations=5000
        )
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        nz = abs(c)
        inlier_pts = np.asarray(remaining.points)[inliers]
        
        if len(inlier_pts) < 1000:
            break
        
        # Calcular area aproximada (bounding box del plano)
        # Proyectar a plano 2D
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u, normal)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        u = u - np.dot(u, normal) * normal
        if np.linalg.norm(u) < 1e-6:
            u = np.array([0.0, 0.0, 1.0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        pts2d = np.column_stack([np.dot(inlier_pts - np.mean(inlier_pts, axis=0), u), 
                                  np.dot(inlier_pts - np.mean(inlier_pts, axis=0), v)])
        area = (pts2d[:,0].max() - pts2d[:,0].min()) * (pts2d[:,1].max() - pts2d[:,1].min())
        
        centroid = np.mean(inlier_pts, axis=0)
        z_mean = centroid[2]
        
        # Clasificar plano
        if nz > 0.85:
            tipo = "horizontal"  # piso o techo
        elif nz < 0.35:
            tipo = "vertical"    # pared
        else:
            tipo = "inclinado"
        
        all_planes.append({
            'id': i,
            'model': plane_model,
            'normal': normal,
            'centroid': centroid,
            'z_mean': z_mean,
            'tipo': tipo,
            'n_points': len(inlier_pts),
            'area': area,
        })
        
        print(f"  Plano {i}: {tipo}, Z={z_mean:.2f}, pts={len(inlier_pts):,}, area={area:.2f}m², nz={nz:.3f}")
        
        remaining = remaining.select_by_index(inliers, invert=True)
    
    # Separar paredes, piso, techo
    horizontales = [p for p in all_planes if p['tipo'] == 'horizontal']
    verticales = [p for p in all_planes if p['tipo'] == 'vertical']
    
    # Filtrar solo planos grandes (> 3m²)
    horizontales = [p for p in horizontales if p['area'] > 3.0]
    verticales = [p for p in verticales if p['area'] > 3.0]
    
    # Identificar piso (Z mas bajo) y techo (Z mas alto)
    piso = None
    techo = None
    if horizontales:
        horizontales.sort(key=lambda x: x['z_mean'])
        piso = horizontales[0]
        if len(horizontales) > 1:
            techo = horizontales[-1]
    
    # Top 4 paredes por area (mayores paredes)
    verticales.sort(key=lambda x: x['area'], reverse=True)
    paredes = verticales[:4]
    
    print(f"\nParedes seleccionadas (area > 3m²): {len(paredes)}")
    print(f"Piso: {'SI' if piso else 'NO'}")
    print(f"Techo: {'SI' if techo else 'NO'}")
    
    # === CALCULAR METRICAS ===
    results = {
        'label': label,
        'n_planes_total': len(all_planes),
        'n_paredes': len(paredes),
        'n_piso': 1 if piso else 0,
        'n_techo': 1 if techo else 0,
        'paredes': [],
        'angulos_adyacentes': [],
        'angulos_opuestas': [],
        'distancias_paredes': [],
        'angulo_pared_suelo': [],
        'dimensiones': None,
    }
    
    # Guardar info de paredes
    for i, p in enumerate(paredes):
        results['paredes'].append({
            'id': p['id'],
            'normal': p['normal'].tolist(),
            'centroid': p['centroid'].tolist(),
            'n_points': p['n_points'],
            'area': p['area'],
        })
    
    # Angulos entre paredes adyacentes (esperado: 90°)
    if len(paredes) >= 2:
        for i in range(len(paredes)):
            for j in range(i+1, len(paredes)):
                n1 = paredes[i]['normal']
                n2 = paredes[j]['normal']
                dot = np.dot(n1, n2)
                # Angulo entre normales
                cos_angle = np.clip(abs(dot), -1.0, 1.0)
                angle = math.degrees(math.acos(cos_angle))
                
                # Si son perpendiculares: dot ~ 0, angle ~ 90°
                # Si son paralelas: dot ~ 1, angle ~ 0°
                if abs(dot) < 0.5:  # Perpendiculares
                    results['angulos_adyacentes'].append({
                        'par': f"Pared{paredes[i]['id']}-Pared{paredes[j]['id']}",
                        'angulo': round(angle, 2),
                        'error_90': round(abs(angle - 90.0), 2),
                        'dot': round(dot, 3),
                    })
                    print(f"  Angulo Pared{paredes[i]['id']}-Pared{paredes[j]['id']}: {angle:.2f}° (error vs 90°: {abs(angle-90.0):.2f}°)")
                else:  # Paralelas
                    results['angulos_opuestas'].append({
                        'par': f"Pared{paredes[i]['id']}-Pared{paredes[j]['id']}",
                        'angulo': round(angle, 2),
                        'error_180': round(abs(angle - 180.0) if dot < 0 else abs(angle - 0.0), 2),
                        'dot': round(dot, 3),
                    })
                    print(f"  Paralelismo Pared{paredes[i]['id']}-Pared{paredes[j]['id']}: {angle:.2f}° (dot={dot:.3f})")
    
    # Distancias entre paredes opuestas
    if len(paredes) >= 2 and piso:
        for i in range(len(paredes)):
            for j in range(i+1, len(paredes)):
                n1 = paredes[i]['normal']
                n2 = paredes[j]['normal']
                dot = np.dot(n1, n2)
                if abs(dot) > 0.5:  # Paralelas
                    # Calcular distancia en 3 alturas
                    z_levels = [
                        piso['z_mean'] + 0.3,
                        piso['z_mean'] + 1.0,
                        piso['z_mean'] + 1.8
                    ]
                    dists = []
                    for z in z_levels:
                        d1 = paredes[i]['model'][3]
                        d2 = paredes[j]['model'][3]
                        n_norm = np.linalg.norm(n1)
                        dist = abs(d1 - d2) / n_norm
                        dists.append(round(dist, 3))
                    
                    results['distancias_paredes'].append({
                        'par': f"Pared{paredes[i]['id']}-Pared{paredes[j]['id']}",
                        'distancias': dists,
                        'variacion': round(max(dists) - min(dists), 3)
                    })
                    print(f"  Distancia Pared{paredes[i]['id']}-Pared{paredes[j]['id']}: {dists} m, variacion: {max(dists)-min(dists):.3f} m")
    
    # Angulo pared-suelo (esperado: 90°)
    if piso and len(paredes) > 0:
        n_piso = piso['normal']
        for p in paredes:
            n_pared = p['normal']
            cos_angle = np.clip(abs(np.dot(n_piso, n_pared)), -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            results['angulo_pared_suelo'].append({
                'pared': f"Pared{p['id']}",
                'angulo': round(angle, 2),
                'error_90': round(abs(angle - 90.0), 2)
            })
            print(f"  Angulo Pared{p['id']}-Suelo: {angle:.2f}° (error vs 90°: {abs(angle-90.0):.2f}°)")
    
    # Calcular dimensiones de la habitacion (bounding box)
    if len(paredes) >= 2 and piso:
        # Distancias entre paredes paralelas
        dists = []
        for i in range(len(paredes)):
            for j in range(i+1, len(paredes)):
                dot = np.dot(paredes[i]['normal'], paredes[j]['normal'])
                if abs(dot) > 0.5:
                    d1 = paredes[i]['model'][3]
                    d2 = paredes[j]['model'][3]
                    n_norm = np.linalg.norm(paredes[i]['normal'])
                    dist = abs(d1 - d2) / n_norm
                    dists.append(dist)
        
        if len(dists) >= 2:
            results['dimensiones'] = {
                'ancho': round(max(dists), 2),
                'largo': round(min(dists), 2),
                'alto': round(abs(techo['z_mean'] - piso['z_mean']), 2) if techo else round(abs(paredes[0]['centroid'][2] - piso['z_mean']), 2),
            }
            print(f"\nDimensiones estimadas: {results['dimensiones']['ancho']}m x {results['dimensiones']['largo']}m x {results['dimensiones']['alto']}m")
    
    return results

# === EJECUTAR ===
all_results = []
for filename, label in INPUT_FILES:
    filepath = INPUT_DIR / filename
    if filepath.exists():
        res = analyze_large_planes(filepath, label)
        if res:
            all_results.append(res)
    else:
        print(f"ERROR: No se encontro {filepath}")

# === GUARDAR RESULTADOS ===
with open(OUTPUT_DIR / "geometry_consistency_v2.txt", "w") as f:
    f.write("="*70 + "\n")
    f.write("ANALISIS DE CONSISTENCIA GEOMETRICA GLOBAL - V2 (nubes originales)\n")
    f.write("="*70 + "\n\n")
    
    for res in all_results:
        f.write(f"--- {res['label']} ---\n")
        f.write(f"Planos detectados: {res['n_planes_total']}\n")
        f.write(f"Paredes (area>3m²): {res['n_paredes']}, Piso: {res['n_piso']}, Techo: {res['n_techo']}\n\n")
        
        if res['dimensiones']:
            f.write(f"Dimensiones estimadas: {res['dimensiones']['ancho']}m x {res['dimensiones']['largo']}m x {res['dimensiones']['alto']}m\n\n")
        
        f.write("ANGULOS ENTRE PAREDES PERPENDICULARES (esperado: 90°):\n")
        for a in res['angulos_adyacentes']:
            f.write(f"  {a['par']}: {a['angulo']}° (error: {a['error_90']}°, dot={a['dot']})\n")
        
        f.write("\nPARALELISMO PAREDES OPUESTAS (esperado: 0° o 180°):\n")
        for a in res['angulos_opuestas']:
            f.write(f"  {a['par']}: {a['angulo']}° (error: {a['error_180']}°, dot={a['dot']})\n")
        
        f.write("\nDISTANCIAS ENTRE PAREDES OPUESTAS:\n")
        for d in res['distancias_paredes']:
            f.write(f"  {d['par']}: {d['distancias']} m, variacion: {d['variacion']} m\n")
        
        f.write("\nANGULOS PARED-SUELO (esperado: 90°):\n")
        for a in res['angulo_pared_suelo']:
            f.write(f"  {a['pared']}: {a['angulo']}° (error: {a['error_90']}°)\n")
        
        f.write("\n" + "="*70 + "\n\n")

print(f"\n{'='*70}")
print(f"RESULTADOS GUARDADOS EN: {OUTPUT_DIR / 'geometry_consistency_v2.txt'}")
print(f"{'='*70}")

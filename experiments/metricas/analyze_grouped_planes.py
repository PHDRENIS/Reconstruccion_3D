#!/usr/bin/env python3
"""
Analisis de Consistencia Geometrica Global - Version 3 (agrupamiento de planos)
Agrupa planos paralelos para identificar las 4 paredes principales de la habitacion.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import math

INPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Posible reconstruccion final")
OUTPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\Metricas Reconstruccion")

INPUT_FILES = [
    ("reconstruction_cloud_cleaned.ply", "Nube Limpia Original"),
    ("reconstruction_cloud.ply", "Nube Original"),
]

def group_planes_by_normals(planes, angle_threshold=15.0):
    """Agrupa planos con normales similares."""
    groups = []
    used = set()
    
    for i, p1 in enumerate(planes):
        if i in used:
            continue
        group = [p1]
        used.add(i)
        
        for j, p2 in enumerate(planes):
            if j in used or i == j:
                continue
            # Calcular angulo entre normales
            dot = np.dot(p1['normal'], p2['normal'])
            angle = math.degrees(math.acos(np.clip(abs(dot), -1.0, 1.0)))
            
            if angle < angle_threshold:
                group.append(p2)
                used.add(j)
        
        # Fusionar grupo: promedio ponderado por puntos
        total_pts = sum(p['n_points'] for p in group)
        avg_normal = np.zeros(3)
        avg_centroid = np.zeros(3)
        for p in group:
            weight = p['n_points'] / total_pts
            avg_normal += p['normal'] * weight
            avg_centroid += p['centroid'] * weight
        
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        # Recalcular d del plano
        d = -np.dot(avg_normal, avg_centroid)
        
        groups.append({
            'model': np.concatenate([avg_normal, [d]]),
            'normal': avg_normal,
            'centroid': avg_centroid,
            'n_points': total_pts,
            'area': sum(p['area'] for p in group),
            'tipo': p1['tipo'],
            'planes': group,
        })
    
    return groups

def analyze_grouped_planes(pcd_path, label):
    print(f"\n{'='*70}")
    print(f"ANALIZANDO: {label}")
    print(f"{'='*70}")
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        print(f"ERROR: No se pudo cargar {pcd_path}")
        return None
    
    print(f"Puntos originales: {len(pcd.points):,}")
    
    pcd_ds = pcd.voxel_down_sample(0.02)
    pts = np.asarray(pcd_ds.points)
    print(f"Puntos despues de voxel: {len(pts):,}")
    
    if len(pts) < 1000:
        print("ERROR: Puntos insuficientes")
        return None
    
    # RANSAC iterativo
    remaining = pcd_ds
    all_planes = []
    MAX_PLANES = 15
    
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
        
        # Calcular area
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
        
        if nz > 0.85:
            tipo = "horizontal"
        elif nz < 0.35:
            tipo = "vertical"
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
    
    # Agrupar planos paralelos
    horizontales = [p for p in all_planes if p['tipo'] == 'horizontal']
    verticales = [p for p in all_planes if p['tipo'] == 'vertical']
    
    h_groups = group_planes_by_normals(horizontales, angle_threshold=20.0)
    v_groups = group_planes_by_normals(verticales, angle_threshold=20.0)
    
    print(f"\nGrupos horizontales: {len(h_groups)}")
    print(f"Grupos verticales: {len(v_groups)}")
    
    # Identificar piso y techo
    piso = None
    techo = None
    if h_groups:
        h_groups.sort(key=lambda x: x['centroid'][2])
        piso = h_groups[0]
        if len(h_groups) > 1:
            techo = h_groups[-1]
    
    # Seleccionar 4 grupos verticales principales (por area)
    v_groups.sort(key=lambda x: x['area'], reverse=True)
    paredes = v_groups[:4]
    
    print(f"\nParedes principales (grupadas): {len(paredes)}")
    print(f"Piso: {'SI' if piso else 'NO'}")
    print(f"Techo: {'SI' if techo else 'NO'}")
    
    # === CALCULAR METRICAS ===
    results = {
        'label': label,
        'n_planes_total': len(all_planes),
        'n_grupos_vertical': len(v_groups),
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
    
    for i, p in enumerate(paredes):
        results['paredes'].append({
            'id': i,
            'normal': p['normal'].tolist(),
            'centroid': p['centroid'].tolist(),
            'n_points': p['n_points'],
            'area': p['area'],
        })
    
    # Angulos entre paredes
    if len(paredes) >= 2:
        for i in range(len(paredes)):
            for j in range(i+1, len(paredes)):
                n1 = paredes[i]['normal']
                n2 = paredes[j]['normal']
                dot = np.dot(n1, n2)
                cos_angle = np.clip(abs(dot), -1.0, 1.0)
                angle = math.degrees(math.acos(cos_angle))
                
                if abs(dot) < 0.5:  # Perpendiculares
                    results['angulos_adyacentes'].append({
                        'par': f"Pared{i}-Pared{j}",
                        'angulo': round(angle, 2),
                        'error_90': round(abs(angle - 90.0), 2),
                        'dot': round(dot, 3),
                    })
                    print(f"  Angulo Pared{i}-Pared{j}: {angle:.2f}° (error vs 90°: {abs(angle-90.0):.2f}°)")
                else:
                    results['angulos_opuestas'].append({
                        'par': f"Pared{i}-Pared{j}",
                        'angulo': round(angle, 2),
                        'error_180': round(abs(angle - 180.0) if dot < 0 else abs(angle - 0.0), 2),
                        'dot': round(dot, 3),
                    })
                    print(f"  Paralelismo Pared{i}-Pared{j}: {angle:.2f}° (dot={dot:.3f})")
    
    # Distancias entre paredes opuestas
    if len(paredes) >= 2 and piso:
        for i in range(len(paredes)):
            for j in range(i+1, len(paredes)):
                n1 = paredes[i]['normal']
                n2 = paredes[j]['normal']
                dot = np.dot(n1, n2)
                if abs(dot) > 0.5:
                    d1 = paredes[i]['model'][3]
                    d2 = paredes[j]['model'][3]
                    n_norm = np.linalg.norm(n1)
                    dist = abs(d1 - d2) / n_norm
                    
                    results['distancias_paredes'].append({
                        'par': f"Pared{i}-Pared{j}",
                        'distancia': round(dist, 3),
                    })
                    print(f"  Distancia Pared{i}-Pared{j}: {dist:.3f} m")
    
    # Angulo pared-suelo
    if piso and len(paredes) > 0:
        n_piso = piso['normal']
        for i, p in enumerate(paredes):
            n_pared = p['normal']
            cos_angle = np.clip(abs(np.dot(n_piso, n_pared)), -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            results['angulo_pared_suelo'].append({
                'pared': f"Pared{i}",
                'angulo': round(angle, 2),
                'error_90': round(abs(angle - 90.0), 2)
            })
            print(f"  Angulo Pared{i}-Suelo: {angle:.2f}° (error vs 90°: {abs(angle-90.0):.2f}°)")
    
    # Dimensiones estimadas
    if len(paredes) >= 2 and piso:
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
                'alto': round(abs(techo['centroid'][2] - piso['centroid'][2]), 2) if techo else round(abs(paredes[0]['centroid'][2] - piso['centroid'][2]), 2),
            }
            print(f"\nDimensiones estimadas: {results['dimensiones']['ancho']}m x {results['dimensiones']['largo']}m x {results['dimensiones']['alto']}m")
    
    return results

# === EJECUTAR ===
all_results = []
for filename, label in INPUT_FILES:
    filepath = INPUT_DIR / filename
    if filepath.exists():
        res = analyze_grouped_planes(filepath, label)
        if res:
            all_results.append(res)
    else:
        print(f"ERROR: No se encontro {filepath}")

# === GUARDAR ===
with open(OUTPUT_DIR / "geometry_consistency_v3.txt", "w") as f:
    f.write("="*70 + "\n")
    f.write("ANALISIS DE CONSISTENCIA GEOMETRICA GLOBAL - V3 (planos agrupados)\n")
    f.write("="*70 + "\n\n")
    
    for res in all_results:
        f.write(f"--- {res['label']} ---\n")
        f.write(f"Planos detectados: {res['n_planes_total']}\n")
        f.write(f"Grupos verticales: {res['n_grupos_vertical']}, Paredes principales: {res['n_paredes']}\n")
        f.write(f"Piso: {res['n_piso']}, Techo: {res['n_techo']}\n\n")
        
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
            f.write(f"  {d['par']}: {d['distancia']} m\n")
        
        f.write("\nANGULOS PARED-SUELO (esperado: 90°):\n")
        for a in res['angulo_pared_suelo']:
            f.write(f"  {a['pared']}: {a['angulo']}° (error: {a['error_90']}°)\n")
        
        f.write("\n" + "="*70 + "\n\n")

print(f"\n{'='*70}")
print(f"RESULTADOS GUARDADOS EN: {OUTPUT_DIR / 'geometry_consistency_v3.txt'}")
print(f"{'='*70}")

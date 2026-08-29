#!/usr/bin/env python3
"""
Genera curvas de aprendizaje de YOLO11x-seg a partir de results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURACION ===
CSV_PATH = Path(r"C:\Users\renea\Desktop\IPN\TT\yolo_segmentation\yolo_ir_finetune\results.csv")
OUTPUT_DIR = Path(r"C:\Users\renea\Desktop\IPN\TT\TT Reporte final\Figuras")

# Cargar datos
df = pd.read_csv(CSV_PATH)
print(f"Datos cargados: {len(df)} epocas")

# Crear figura grande (2 filas x 2 columnas)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Curvas de Aprendizaje - YOLO11x-seg Fine-Tuning (Infrarrojo)', fontsize=14, fontweight='bold')

# --- Subplot 1: Loss de Segmentacion ---
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train/seg_loss'], 'b-', label='Train Seg Loss', linewidth=2)
ax1.plot(df['epoch'], df['val/seg_loss'], 'r-', label='Val Seg Loss', linewidth=2, alpha=0.7)
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss de Segmentacion')
ax1.set_title('Loss de Segmentacion (Train vs Val)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Loss de Clasificacion ---
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train CLS Loss', linewidth=2)
ax2.plot(df['epoch'], df['val/cls_loss'], 'r-', label='Val CLS Loss', linewidth=2, alpha=0.7)
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Loss de Clasificacion')
ax2.set_title('Loss de Clasificacion (Train vs Val)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Subplot 3: mAP50 (Bounding Box y Mascara) ---
ax3 = axes[1, 0]
ax3.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', label='mAP50 (Box)', linewidth=2)
ax3.plot(df['epoch'], df['metrics/mAP50(M)'], 'm-', label='mAP50 (Mask)', linewidth=2)
ax3.set_xlabel('Epoca')
ax3.set_ylabel('mAP50 (%)')
ax3.set_title('mAP50: Bounding Box vs Mascara')
ax3.legend()
ax3.grid(True, alpha=0.3)
# Anotar valor final
ax3.axhline(y=df['metrics/mAP50(M)'].iloc[-1], color='m', linestyle='--', alpha=0.3)
ax3.text(1, df['metrics/mAP50(M)'].iloc[-1]+0.02, f"Final: {df['metrics/mAP50(M)'].iloc[-1]:.3f}", color='m')

# --- Subplot 4: mAP50-95 (Bounding Box y Mascara) ---
ax4 = axes[1, 1]
ax4.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'g-', label='mAP50-95 (Box)', linewidth=2)
ax4.plot(df['epoch'], df['metrics/mAP50-95(M)'], 'm-', label='mAP50-95 (Mask)', linewidth=2)
ax4.set_xlabel('Epoca')
ax4.set_ylabel('mAP50-95 (%)')
ax4.set_title('mAP50-95: Bounding Box vs Mascara')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'yolo_training_curves.png', dpi=300, bbox_inches='tight')
print(f"Figura guardada: {OUTPUT_DIR / 'yolo_training_curves.png'}")

# === FIGURA ADICIONAL: Learning Rate ===
fig2, ax5 = plt.subplots(figsize=(10, 4))
ax5.semilogy(df['epoch'], df['lr/pg0'], 'k-', label='LR Backbone', linewidth=2)
ax5.semilogy(df['epoch'], df['lr/pg1'], 'b-', label='LR Intermedio', linewidth=2)
ax5.semilogy(df['epoch'], df['lr/pg2'], 'r-', label='LR Cabeza', linewidth=2)
ax5.set_xlabel('Epoca')
ax5.set_ylabel('Learning Rate (log scale)')
ax5.set_title('Evolucion del Learning Rate (AdamW)')
ax5.legend()
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'yolo_learning_rate.png', dpi=300, bbox_inches='tight')
print(f"Figura guardada: {OUTPUT_DIR / 'yolo_learning_rate.png'}")

print("\n=== RESUMEN DE VALORES FINALES ===")
print(f"mAP50 (Box): {df['metrics/mAP50(B)'].iloc[-1]:.4f}")
print(f"mAP50 (Mask): {df['metrics/mAP50(M)'].iloc[-1]:.4f}")
print(f"mAP50-95 (Box): {df['metrics/mAP50-95(B)'].iloc[-1]:.4f}")
print(f"mAP50-95 (Mask): {df['metrics/mAP50-95(M)'].iloc[-1]:.4f}")
print(f"Val Seg Loss final: {df['val/seg_loss'].iloc[-1]:.4f}")
print(f"Val CLS Loss final: {df['val/cls_loss'].iloc[-1]:.4f}")

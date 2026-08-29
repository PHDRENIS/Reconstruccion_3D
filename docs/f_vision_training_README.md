# Fine-Tuning YOLO11s-seg para SUN-RGB-D

Este módulo permite entrenar un modelo YOLO con segmentación específicamente para escenas de interiores (SUN-RGB-D).

## Estructura del Proyecto

```
F Vision/
├── train/
│   ├── prepare_data.py      # Prepara datos para YOLO
│   ├── train.py            # Entrena el modelo
│   └── evaluate.py         # Evalúa el modelo
│
├── data/
│   └── sunrgbd/           # Dataset preparado
│       ├── images/
│       ├── labels/
│       └── sunrgbd.yaml
│
├── models/
│   ├── yolo11s-seg.pt     # Modelo base
│   └── yolo11s-sunrgbd-seg.pt  # Modelo entrenado (resultado)
│
└── runs/                   # Resultados del entrenamiento
```

## Clases de SUN-RGB-D (17 clases)

| ID | Clase |
|----|-------|
| 0 | bed |
| 1 | chair |
| 2 | table |
| 3 | sofa |
| 4 | desk |
| 5 | dresser |
| 6 | bookshelf |
| 7 | lamp |
| 8 | pillow |
| 9 | sink |
| 10 | bathtub |
| 11 | toilet |
| 12 | box |
| 13 | counter |
| 14 | refrigerator |
| 15 | tv |
| 16 | curtain |

## Pasos para el Entrenamiento

### 1. Instalar dependencias

```bash
cd C:\Users\renea\Desktop\TT\F Vision

# Crear entorno virtual (si no existe)
python -m venv venv
venv\Scripts\activate

# Instalar PyTorch con CUDA (si tienes GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Instalar Ultralytics
pip install ultralytics
```

### 2. Preparar los datos

```bash
python train/prepare_data.py
```

Este script:
- Lee las imágenes RGB y máscaras de `SUNRBG_IMAGES/`
- Convierte las máscaras al formato YOLO segmentación
- Divide en train (85%) y validation (15%)
- Crea el archivo `sunrgbd.yaml`

### 3. Entrenar el modelo

```bash
python train/train.py
```

Configuración en `train_config.yaml`:
- **Épocas**: 100
- **Modelo base**: YOLO11s-seg
- **Batch size**: 8 (ajustar según VRAM)
- **Tamaño imagen**: 640

### 4. Evaluar el modelo

```bash
python train/evaluate.py
```

### 5. Usar el modelo entrenado

Una vez entrenado, el modelo se guardará en:
```
models/yolo11s-sunrgbd-seg.pt
```

El pipeline de procesamiento lo usará automáticamente si está disponible.

## Uso del Pipeline con Modelo Entrenado

El pipeline detecta automáticamente el modelo entrenado:

1. Busca `models/yolo11s-sunrgbd-seg.pt`
2. Si existe, lo usa (17 clases de interior)
3. Si no existe, usa el modelo pre-entrenado (80 clases COCO)

```bash
python src/main.py
```

## Requisitos del Sistema

- **GPU**: Recomendado (8GB+ VRAM para YOLO11s-seg)
- **CPU**: Funciona pero es más lento
- **RAM**: 16GB+
- **Espacio en disco**: ~10GB para dataset y resultados

## Tiempo Estimado

| Paso | Tiempo |
|------|--------|
| Preparar datos | 15-30 min |
| Entrenamiento (100 épocas, GPU) | 4-8 horas |
| Evaluación | 15-30 min |

## Solución de Problemas

### Error: "CUDA out of memory"
Reduce el batch size en `train_config.yaml`:
```yaml
TRAINING:
  batch: 4  # Reducir de 8 a 4
```

### Error: "No se encontró modelo"
Asegúrate de ejecutar primero:
```bash
python train/prepare_data.py
python train/train.py
```

## Notas

- El modelo entrenado usará las 17 clases de interiores de SUN-RGB-D
- Las detecciones serán más precisas para escenas de interior
- El pipeline guardará máscaras por clase automáticamente

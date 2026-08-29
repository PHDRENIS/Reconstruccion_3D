# FusiónVisión - Reconstrucción 3D con Fusión Semántica

Este proyecto implementa el pipeline de procesamiento semántico y segmentación de instancias descrito en las secciones 4.7 y 4.8 del documento de tesis.

## Estructura del Proyecto

```
F Vision/
├── src/
│   ├── data/           # Cargador de datos SUN-RGB-D
│   ├── models/         # Modelos de detección y segmentación
│   ├── processing/      # Pipeline de procesamiento semántico
│   ├── utils/          # Utilidades
│   └── evaluation/    # Métricas de evaluación
├── models/             # Modelos pre-entrenados
└── output/             # Resultados generados
```

## Requisitos

- Python 3.10+
- GPU con CUDA (recomendado)
- Ver requirements.txt

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python src/main.py
```

## Modelos Utilizados

- YOLO11x: Detección rápida de objetos
- RT-DETRv2: Detección de precisión
- FastSAM: Segmentación de instancias

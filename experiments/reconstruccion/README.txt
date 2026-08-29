================================================================================
           PIPELINE DE RECONSTRUCCION 3D V1 — LISTO PARA REPLICAR
================================================================================

ARCHIVOS:
  run_final_reconstruction.py    - Pipeline principal (YOLO IR + ICP + TSDF)
  run_ir_depth_pipeline.py       - Pipeline alternativo (depth completion + ICP)
  run_reconstruction.bat         - Comando de ejecucion
  README.txt                     - Este archivo

DEPENDENCIAS:
  Python 3.12 con:
    open3d 0.19.0
    ultralytics 8.x
    opencv-python 4.x
    numpy, scipy

MODELO REQUERIDO:
  C:\Users\renea\Desktop\IPN\TT\yolo_segmentation\yolo_ir_finetune\weights\best.pt
  (YOLO11x-seg fine-tuneado para IR, nc=1, mAP50=0.890)

DATOS DE ENTRADA:
  C:\Users\renea\Desktop\IPN\TT\F Vision\Reentreno\images\yolo_data_*.jpg   (319 frames IR)
  C:\Users\renea\Desktop\IPN\TT\F Vision\Reentreno\depth_maps\yolo_data_*.png (319 frames Depth)

COMANDO DE EJECUCION (copiar y pegar en terminal desde C:\Users\renea\Desktop\IPN\TT\TT\FV2):

  .venv312\Scripts\python.exe "C:\Users\renea\Desktop\IPN\TT\Nueva reconstruccion\run_final_reconstruction.py" --ir-dir "C:\Users\renea\Desktop\IPN\TT\F Vision\Reentreno\images" --depth-dir "C:\Users\renea\Desktop\IPN\TT\F Vision\Reentreno\depth_maps" --output-dir "C:\Users\renea\Desktop\IPN\TT\Nueva reconstruccion" --yolo-model "C:\Users\renea\Desktop\IPN\TT\yolo_segmentation\yolo_ir_finetune\weights\best.pt" --depth-scale 0.001 --voxel-length 0.008 --sdf-trunc 0.03 --yolo-conf 0.3

METRICAS ESPERADAS:
  frames: 319
  avg_yolo_dets: ~15.4
  avg_mask_ratio: ~0.57
  avg_valid_depth: ~0.92
  icp_fitness: ~0.97
  icp_rmse: ~0.026
  mesh_vertices: ~5,600,000
  mesh_triangles: ~10,000,000

SALIDA:
  reconstruction_mesh.ply     - Malla 3D (398 MB)
  reconstruction_cloud.ply    - Nube de puntos (260 MB)
  evaluation.txt              - Metricas

PIPELINE:
  1) Carga 319 pares IR + Depth
  2) YOLO IR: segmenta objetos (15 detecciones promedio)
  3) ICP frame-a-frame:
     - Init: RGBD Odometry (jacobiano hibrido)
     - Refinamiento: ICP Point-to-Plane (dist=0.08m, iter=60)
  4) TSDF Fusion:
     - Voxel: 8 mm
     - SDF trunc: 3 cm
     - Color: IR replicado a BGR
  5) Extraccion de malla (Marching Cubes)
  6) Exportacion PLY
================================================================================

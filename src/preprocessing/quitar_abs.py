import os

#Used only to rename files by removing '_abs' from their names in specified directories, they were added by mistake on previous steps.

DIRECTORIOS_A_CORREGIR = [
    "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\mask\\",
    "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Train\\rgb\\",
    "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Validation\\mask\\",
    "C:\\Users\\victo\\OneDrive\\Documents\\TT\\SUNRBG_IMAGES\\Validation\\rgb\\"
]

TEXTO_A_ELIMINAR = "_abs"


def corregir_nombres():
    print(f"Buscando archivos con '{TEXTO_A_ELIMINAR}' para renombrar...\n")
    
    total_renombrados = 0

    for carpeta in DIRECTORIOS_A_CORREGIR:
        if not os.path.exists(carpeta):
            print(f"[ADVERTENCIA] No existe la carpeta: {carpeta}")
            continue

        print(f" Analizando: {carpeta}")
        archivos = os.listdir(carpeta)
        contador_carpeta = 0

        for archivo in archivos:
            if TEXTO_A_ELIMINAR in archivo:
                # Construir rutas
                ruta_vieja = os.path.join(carpeta, archivo)
                
                # Crear nuevo nombre (reemplazando  por nada)
                nuevo_nombre = archivo.replace(TEXTO_A_ELIMINAR, "")
                ruta_nueva = os.path.join(carpeta, nuevo_nombre)

            
                try:
                    os.rename(ruta_vieja, ruta_nueva)
                    
                    contador_carpeta += 1
                except FileExistsError:
                    print(f"    [ERROR] No se pudo renombrar {archivo} porque {nuevo_nombre} YA EXISTE.")
                except Exception as e:
                    print(f"  [ERROR] Falló {archivo}: {e}")

        print(f"    Corregidos en esta carpeta: {contador_carpeta}")
        total_renombrados += contador_carpeta

    print(f"\n Se renombraron un total de {total_renombrados} archivos.")
    print("Intenta correr train.py de nuevo.")

if __name__ == "__main__":
    corregir_nombres()
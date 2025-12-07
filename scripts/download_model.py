"""
Script para descargar modelos desde Google Drive.
"""

from pathlib import Path
import shutil

def download_from_drive(model_name: str, drive_path: str = None):
    """
    Descarga un modelo desde Google Drive a local.
    
    Args:
        model_name: Nombre del modelo
        drive_path: Ruta en Drive (None = usar default)
    """
    if drive_path is None:
        drive_path = f"/content/drive/MyDrive/SignLanguageTranslator/models/{model_name}"
    
    local_path = Path("models") / model_name
    
    try:
        print(f"📥 Descargando: {model_name}")
        print(f"   Desde: {drive_path}")
        print(f"   Hacia: {local_path}")
        
        shutil.copy(drive_path, local_path)
        print("✅ Descarga completada")
        
    except FileNotFoundError:
        print(f"❌ Error: Modelo no encontrado en {drive_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        download_from_drive(model_name)
    else:
        print("Uso: python download_model.py <model_name.h5>")
"""
Ejemplo básico de captura de datos.
Muestra cómo capturar muestras de señas.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from key_points_extractor import KeyPointsExtractor
from config import DATA_DIR

def capture_sign(sign_name: str, num_samples: int = 10, frames_per_sample: int = 30):
    """
    Captura muestras de una seña.
    
    Args:
        sign_name: Nombre de la seña
        num_samples: Número de muestras a capturar
        frames_per_sample: Frames por muestra
    """
    # Crear directorio para la seña
    sign_dir = DATA_DIR / "raw" / sign_name
    sign_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar extractor y cámara
    extractor = KeyPointsExtractor()
    cap = cv2.VideoCapture(0)
    
    print(f"📹 Capturando {num_samples} muestras de '{sign_name}'")
    print(f"   {frames_per_sample} frames por muestra")
    print("\nPresiona 'SPACE' para iniciar captura")
    print("Presiona 'Q' para salir\n")
    
    sample_count = 0
    
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame
        processed_frame, _ = extractor.extract_keypoints(frame)
        
        # Mostrar instrucciones
        cv2.putText(processed_frame, f"Seña: {sign_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Muestra: {sample_count + 1}/{num_samples}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, "Presiona SPACE para capturar", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Captura de Señas', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Iniciar captura
            print(f"Capturando muestra {sample_count + 1}...")
            sequence = []
            
            for frame_num in range(frames_per_sample):
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, features = extractor.extract_keypoints(frame)
                sequence.append(features)
                
                # Mostrar progreso
                cv2.putText(processed_frame, f"Capturando: {frame_num + 1}/{frames_per_sample}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Captura de Señas', processed_frame)
                cv2.waitKey(33)  # ~30 FPS
            
            # Guardar muestra
            sequence_array = np.array(sequence)
            sample_file = sign_dir / f"sample_{sample_count:03d}.npy"
            np.save(sample_file, sequence_array)
            
            print(f"✓ Guardada: {sample_file}")
            sample_count += 1
            
            extractor.reset_history()
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.release()
    
    print(f"\n✅ Captura completada: {sample_count} muestras de '{sign_name}'")

if __name__ == "__main__":
    # Ejemplo de uso
    capture_sign("hola", num_samples=10, frames_per_sample=30)
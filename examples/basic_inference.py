"""
Ejemplo básico de inferencia.
Muestra cómo usar un modelo entrenado para hacer predicciones.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from key_points_extractor import KeyPointsExtractor
from config import MODELS_DIR, DATA_DIR
from tensorflow import keras

def predict_realtime(model_path: str = None):
    """
    Predicción en tiempo real.
    
    Args:
        model_path: Ruta al modelo (None = usar el mejor)
    """
    # Cargar modelo
    if model_path is None:
        model_path = MODELS_DIR / "sign_language_model_best.h5"
    
    print(f"📥 Cargando modelo: {model_path}")
    model = keras.models.load_model(str(model_path))
    
    # Cargar sign_map
    sign_map_path = DATA_DIR / "sign_map.json"
    if sign_map_path.exists():
        with open(sign_map_path) as f:
            sign_map = json.load(f)
    else:
        sign_map = {}
    
    # Inicializar
    extractor = KeyPointsExtractor()
    cap = cv2.VideoCapture(0)
    
    print("🌐 Traducción en tiempo real")
    print("Presiona 'Q' para salir\n")
    
    sequence_buffer = []
    sequence_length = 30
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extraer features
        processed_frame, features = extractor.extract_keypoints(frame)
        
        # Agregar a buffer
        sequence_buffer.append(features)
        if len(sequence_buffer) > sequence_length:
            sequence_buffer.pop(0)
        
        # Hacer predicción si tenemos suficientes frames
        if len(sequence_buffer) == sequence_length:
            # Preparar secuencia
            sequence = np.array(sequence_buffer).reshape(1, sequence_length, 240)
            
            # Predecir
            prediction = model.predict(sequence, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Obtener nombre de la seña
            predicted_sign = sign_map.get(str(predicted_class), f"Clase {predicted_class}")
            
            # Dibujar predicción
            cv2.putText(processed_frame, f"Seña: {predicted_sign}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Confianza: {confidence:.2%}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Traducción en Tiempo Real', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.release()

if __name__ == "__main__":
    predict_realtime()
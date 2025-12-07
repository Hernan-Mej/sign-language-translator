"""
Ejemplo básico de entrenamiento.
Muestra cómo entrenar un modelo con datos capturados.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training import Trainer
from config import DATA_DIR, MODELS_DIR, LOGS_DIR

def train_model():
    """Entrena un modelo básico."""
    
    print("="*60)
    print("Entrenamiento de Modelo - Sign Language Translator")
    print("="*60)
    
    # Inicializar trainer
    trainer = Trainer(
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        logs_dir=LOGS_DIR,
        use_augmentation=True
    )
    
    # Cargar datos
    print("\n📊 Cargando datos...")
    X, y, sign_map = trainer.load_and_prepare_data()
    
    # Normalizar secuencias
    print("🔧 Normalizando secuencias...")
    X = trainer.normalize_sequence_lengths(X, target_length=30)
    
    # Split datos
    print("✂️ Dividiendo datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Augmentar
    print("🔄 Aplicando data augmentation...")
    X_train_aug, y_train_aug = trainer.augment_data(X_train, y_train)
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    model, history = trainer.train_model(
        X_train_aug, y_train_aug,
        X_val, y_val,
        num_classes=len(sign_map),
        model_name="sign_language_model"
    )
    
    # Evaluar
    print("\n📈 Evaluando modelo...")
    results = trainer.evaluate_model(model, X_test, y_test, sign_map)
    
    # Resultados
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\n📊 Resultados:")
    print(f"   Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"   Top-3 Accuracy: {results['metrics'].get('top_3_accuracy', 0):.2%}")
    print(f"\n💾 Modelo guardado en: {MODELS_DIR}")
    print("="*60)

if __name__ == "__main__":
    train_model()
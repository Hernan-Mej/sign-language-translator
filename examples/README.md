# 📚 Examples

Ejemplos prácticos de uso del sistema.

## 🎯 Ejemplos Disponibles

### 1. basic_capture.py
Captura de datos para entrenamiento.
```bash
python examples/basic_capture.py
```

**Uso:**
- Presiona SPACE para iniciar captura
- Realiza la seña frente a la cámara
- Presiona Q para salir

### 2. basic_training.py
Entrenamiento completo del modelo.
```bash
python examples/basic_training.py
```

**Proceso:**
1. Carga datos de `data/raw/`
2. Aplica data augmentation
3. Entrena modelo
4. Guarda en `models/`

### 3. basic_inference.py
Predicción en tiempo real.
```bash
python examples/basic_inference.py
```

**Uso:**
- Realiza señas frente a la cámara
- El sistema traduce en tiempo real
- Presiona Q para salir

## 📝 Modificar Ejemplos

Puedes personalizar estos ejemplos editando los parámetros:
```python
# En basic_capture.py
capture_sign("hola", num_samples=15, frames_per_sample=40)

# En basic_training.py
trainer = Trainer(use_augmentation=False)  # Sin augmentation

# En basic_inference.py
predict_realtime(model_path="models/my_custom_model.h5")
```
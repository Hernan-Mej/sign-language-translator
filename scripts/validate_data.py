"""
Script para validar el dataset.
"""

import numpy as np
from pathlib import Path
import json

def validate_dataset(data_dir: str = "data/raw"):
    """Valida la estructura y calidad del dataset."""
    
    data_path = Path(data_dir)
    
    print("="*60)
    print("Validación del Dataset")
    print("="*60)
    
    sign_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"\n📊 Señas encontradas: {len(sign_dirs)}")
    
    total_samples = 0
    issues = []
    
    for sign_dir in sorted(sign_dirs):
        samples = list(sign_dir.glob("*.npy"))
        num_samples = len(samples)
        total_samples += num_samples
        
        print(f"\n📁 {sign_dir.name}:")
        print(f"   Muestras: {num_samples}")
        
        # Validar cada muestra
        for sample in samples:
            try:
                data = np.load(sample)
                
                # Verificar dimensiones
                if len(data.shape) != 2:
                    issues.append(f"{sample}: Dimensión incorrecta {data.shape}")
                elif data.shape[1] != 240:
                    issues.append(f"{sample}: Features incorrectas ({data.shape[1]} != 240)")
                elif data.shape[0] < 10:
                    issues.append(f"{sample}: Muy corta ({data.shape[0]} frames)")
                    
            except Exception as e:
                issues.append(f"{sample}: Error al cargar - {e}")
        
        # Recomendaciones
        if num_samples < 10:
            print(f"   ⚠️  Recomendado: Al menos 10 muestras")
        elif num_samples < 15:
            print(f"   ✓  Aceptable (recomendado: 15+)")
        else:
            print(f"   ✅ Excelente")
    
    print(f"\n📊 Resumen:")
    print(f"   Total de señas: {len(sign_dirs)}")
    print(f"   Total de muestras: {total_samples}")
    print(f"   Promedio por seña: {total_samples / len(sign_dirs) if sign_dirs else 0:.1f}")
    
    if issues:
        print(f"\n⚠️  Problemas encontrados:")
        for issue in issues[:10]:  # Mostrar solo los primeros 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... y {len(issues) - 10} más")
    else:
        print("\n✅ No se encontraron problemas")
    
    print("="*60)

if __name__ == "__main__":
    validate_dataset()
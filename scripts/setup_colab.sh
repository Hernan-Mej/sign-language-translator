#!/bin/bash

# Setup script for Google Colab
# This script automates the setup process

echo "======================================"
echo "Sign Language Translator - Colab Setup"
echo "======================================"

# Mount Google Drive
echo ""
echo "📁 Mounting Google Drive..."
python3 << EOF
from google.colab import drive
drive.mount('/content/drive')
print("✓ Drive mounted")
EOF

# Clone repository
echo ""
echo "📥 Cloning repository..."
cd /content
if [ -d "sign-language-translator" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd sign-language-translator
    git pull
else
    git clone https://github.com/YOUR_USERNAME/sign-language-translator.git
    cd sign-language-translator
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Add to Python path
echo ""
echo "🔧 Configuring Python path..."
python3 << EOF
import sys
sys.path.insert(0, '/content/sign-language-translator/src')
print("✓ Python path configured")
EOF

# Create Drive directories
echo ""
echo "📂 Creating Drive directories..."
python3 << EOF
from pathlib import Path
import os

drive_root = Path('/content/drive/MyDrive')
project_dir = drive_root / 'SignLanguageTranslator'

dirs = [
    project_dir / 'data' / 'raw',
    project_dir / 'data' / 'processed',
    project_dir / 'models',
    project_dir / 'logs'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"✓ {dir_path}")

print("✓ Directory structure created")
EOF

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Run the Colab notebook"
echo "2. Start capturing data"
echo "3. Train your model"
echo ""
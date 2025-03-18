#!/bin/bash

# Ім'я вихідного коду та виконуваного файлу
SRC_FILE="matrix_mul.cu"
OUTPUT_FILE="matrix_mul"

# Перевірка наявності CUDA-компілятора nvcc
if ! command -v nvcc &> /dev/null
then
    echo "❌ Помилка: CUDA-компілятор (nvcc) не знайдено. Переконайтесь, що CUDA встановлено."
    exit 1
fi

# Компіляція CUDA-програми
echo "🚀 Компіляція $SRC_FILE..."
nvcc -O3 -arch=sm_50 -o $OUTPUT_FILE $SRC_FILE

# Перевірка, чи компіляція була успішною
if [ $? -ne 0 ]; then
    echo "❌ Помилка компіляції!"
    exit 1
fi

echo "✅ Компіляція успішна!"

# Запуск програми
echo "🎯 Запуск CUDA-множення матриць..."
./$OUTPUT_FILE

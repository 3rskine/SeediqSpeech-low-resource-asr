#!/bin/bash
# 目標資料夾
DIRS=("data/train_mixed" "data/seediq_dev" "data/seediq_test")

for dir in "${DIRS[@]}"; do
    if [ -f "$dir/text" ]; then
        echo "🔠 正在轉換 $dir/text 為全大寫..."
        cp "$dir/text" "$dir/text.bak" # 備份
        tr '[:lower:]' '[:upper:]' < "$dir/text.bak" > "$dir/text"
    fi
done
echo "✅ 所有文本已轉換為大寫！"

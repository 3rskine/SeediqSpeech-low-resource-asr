#!/usr/bin/env bash
# 設定 bash 嚴格模式
set -e
set -u
set -o pipefail

# 1. 資料夾設定
train_set=seediq_train
valid_set=seediq_dev
test_sets="seediq_dev seediq_test"

# 2. 設定檔路徑
asr_config=conf/train_asr_xlsr_lora.yaml
inference_config=conf/decode_asr.yaml

# 3. 啟動訓練
# 修正重點：將 --stage 改為 3 (從資料格式化開始)
# 這樣系統才會建立 dump/ 資料夾，解決 No such file 錯誤
./asr.sh \
    --lang seediq \
    --ngpu 1 \
    --stage 3 \
    --stop_stage 11 \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_train_text "data/${train_set}/text" \
    --token_type char \
    --asr_tag "seediq_xlsr_lora_mvp"

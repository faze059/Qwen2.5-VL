#!/bin/bash
# Qwen2.5-VL 72B LoRA Training Script (Mac Studio 512GB)
# Dry-run: max_steps 5 for quick sanity-check on dataset

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Base model & dataset
llm=Qwen/Qwen2.5-VL-72B-Instruct
datasets="protocol_tables"

# LoRA config
lora_r=16
lora_alpha=32
lora_dropout=0.05

# Hyper-params
lr=5e-5
batch_size=1
grad_accum_steps=16

entry_file=qwen-vl-finetune/qwenvl/train/train_qwen.py
output_dir=./output_72b_lora
run_name="qwen2vl-72b-lora-mac"

args="
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --use_lora True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps 5 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 28672 \
    --min_pixels 784 \
    --learning_rate ${lr} \
    --save_strategy no \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --run_name ${run_name} \
    --report_to none \
    --remove_unused_columns False"

echo "🚀 Starting Qwen2.5-VL-72B LoRA dry-run (5 steps)..."
echo "Model: ${llm}"
python ${entry_file} ${args}

echo "✅ Dry-run finished. Check logs above." 
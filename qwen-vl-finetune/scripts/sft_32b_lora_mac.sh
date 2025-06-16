#!/bin/bash
# Mac Studio M3 Ultra - 32B LoRA Training Script
# Optimized for 128GB GPU + 512GB Unified Memory

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Model configuration  
llm=Qwen/Qwen2.5-VL-32B-Instruct

# LoRA rank/alpha
lora_r=32
lora_alpha=16

# LoRA hyperparameters - optimized for 32B
lr=1e-4
batch_size=1
grad_accum_steps=16

# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration - using demo data for testing
datasets="single_images"

# Output configuration
run_name="qwen2vl-32b-lora-mac"
output_dir=./output_32b_lora

# Training arguments optimized for Mac Studio
args="
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --use_lora True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.25 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 28672 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name} \
    --report_to none \
    --remove_unused_columns False"

echo "🚀 Starting 32B LoRA training on Mac Studio..."
echo "Model: ${llm}"
echo "LoRA r=${lora_r}, alpha=${lora_alpha}"
echo "Batch size: ${batch_size}, Accumulation: ${grad_accum_steps}"
echo "Learning rate: ${lr}"

# Launch training (single GPU on Mac)
python ${entry_file} ${args}

echo "✅ Training completed!"
echo "Model saved to: ${output_dir}" 
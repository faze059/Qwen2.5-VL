#!/bin/bash
# Mac Studio - 7B LoRA Testing Script
# Quick test to verify LoRA training works before trying 32B

# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Model configuration - using 7B for quick testing
llm=Qwen/Qwen2.5-VL-7B-Instruct

# LoRA rank/alpha
lora_r=16
lora_alpha=8

# LoRA hyperparameters
lr=2e-4
batch_size=2
grad_accum_steps=4

# Training entry point
entry_file=qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration - using demo data
datasets="single_images"

# Output configuration
run_name="qwen2vl-7b-lora-test"
output_dir=./output_7b_lora_test

# Training arguments - minimal setup for testing
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
    --num_train_epochs 0.1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 12544 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --run_name ${run_name} \
    --report_to none \
    --remove_unused_columns False"

echo "🧪 Starting 7B LoRA test training..."
echo "Model: ${llm}"
echo "Dataset: ${datasets}"
echo "LoRA r=${lora_r}, alpha=${lora_alpha}"
echo "This is a quick test - should take 2-3 minutes"

# Launch training
python ${entry_file} ${args}

echo "✅ Test completed!"
echo "If successful, you can try 32B training with: ./scripts/sft_32b_lora_mac.sh" 
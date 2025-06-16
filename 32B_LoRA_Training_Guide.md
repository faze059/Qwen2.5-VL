# 🚀 Qwen2.5-VL 32B LoRA Training Guide for Mac Studio

This guide walks you through training a **32B Qwen2.5-VL model with LoRA (Low-Rank Adaptation)** on Mac Studio M3 Ultra, with **frozen vision encoder** for optimal memory usage.

## 📋 Prerequisites

### Hardware Requirements
- **Mac Studio M3 Ultra** (24-core CPU, 60-core GPU)
- **128GB GPU Memory + 512GB Unified Memory**
- At least **200GB free storage** for model and checkpoints

### Software Requirements
- macOS 14.0+ (Sonoma)
- Python 3.9+
- PyTorch 2.0+ with MPS support

## 🛠️ Setup Instructions

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd qwen-vl-finetune

# Install LoRA training dependencies
pip install -r ../requirements_lora.txt

# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### Step 2: Prepare Your Dataset

Your dataset should be in JSON format with the following structure:

```json
[
    {
        "id": "sample_1",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat do you see in this image?"
            },
            {
                "from": "gpt", 
                "value": "I can see a beautiful landscape with mountains and a lake."
            }
        ],
        "image": "path/to/your/image1.jpg"
    }
]
```

## 🔧 Training Configuration

### Key LoRA Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| `lora_r` | 32 | LoRA rank (higher = more parameters) |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_dropout` | 0.1 | Dropout rate for LoRA layers |
| `target_modules` | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj | Which linear layers to apply LoRA |

### Memory Optimization Settings

| Parameter | Value | Description |
|-----------|--------|-------------|
| `batch_size` | 1 | Per-device batch size |
| `grad_accum_steps` | 16 | Gradient accumulation (effective batch = 16) |
| `gradient_checkpointing` | True | Trade compute for memory |
| `tune_mm_vision` | False | Freeze vision encoder |
| `bf16` | True | Use bfloat16 precision |

## 🚀 Training Commands

### Quick Start Training

```bash
# Edit the dataset path in the script first
nano scripts/sft_32b_lora_mac.sh

# Modify this line with your dataset:
datasets="your_dataset1,your_dataset2"

# Start training
./scripts/sft_32b_lora_mac.sh
```

### Custom Training Command

```bash
python qwenvl/train/train_qwen.py \
    --model_name_or_path Qwen/Qwen2.5-VL-32B-Instruct \
    --dataset_use your_dataset_name \
    --data_flatten True \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ./output_32b_lora \
    --num_train_epochs 0.25 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --gradient_checkpointing True \
    --save_steps 100 \
    --logging_steps 1 \
    --report_to none
```

## 📊 Expected Performance

### Training Speed
- **~20-30 seconds per step** on Mac Studio M3 Ultra
- **Memory usage**: ~80-90GB GPU + additional unified memory
- **Total training time**: 4-8 hours for 0.25 epoch on 10K images

### Resource Monitoring

```bash
# Monitor GPU memory usage
sudo powermetrics --samplers gpu_power -n 1 --hide-cpu-duty-cycle

# Monitor system memory
vm_stat
```

## 🧪 Testing Your Trained Model

### Load and Test LoRA Model

```bash
python scripts/inference_lora.py \
    --base_model Qwen/Qwen2.5-VL-32B-Instruct \
    --lora_adapter ./output_32b_lora \
    --image path/to/test_image.jpg \
    --prompt "Describe this image in detail."
```

### Merge LoRA Back to Base Model (Optional)

```python
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

# Load base model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

# Load LoRA model
lora_model = PeftModel.from_pretrained(base_model, "./output_32b_lora")

# Merge and save
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./merged_32b_model")
```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce batch size or gradient accumulation
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
```

**2. MPS Fallback Warnings**
```bash
# This is normal - some operations fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**3. Slow Training Speed**
```bash
# Reduce sequence length if your data allows
--model_max_length 2048
```

**4. Model Loading Issues**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
```

### Performance Tips

1. **Close other applications** to free up memory
2. **Use SSD storage** for faster I/O
3. **Monitor temperature** - training will throttle if too hot
4. **Use smaller images** if possible (max_pixels parameter)

## 📁 Output Structure

After training completes, you'll find:

```
output_32b_lora/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin        # LoRA weights
├── training_args.bin        # Training arguments
├── trainer_state.json       # Training state
├── checkpoint-100/          # Intermediate checkpoints
├── checkpoint-200/
└── ...
```

## ⚡ Next Steps

1. **Evaluate your model** on validation data
2. **Try different LoRA ranks** (r=16, 64) for comparison
3. **Experiment with learning rates** (5e-5, 2e-4)
4. **Scale up to full training** if results look promising

## 🎯 Expected Results

With proper training data:
- **Loss should decrease** from ~2.5 to ~1.0 or lower
- **Visual question answering** should improve noticeably
- **LoRA adapters** are only ~100-200MB vs 64GB full model

---

**Happy Training! 🎉**

For issues or questions, check the training logs in your output directory. 
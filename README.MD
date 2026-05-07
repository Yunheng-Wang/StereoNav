<p align="center">
  <h2 align="center"> What Limits Vision-and-Language Navigation? </h2>
  <p align="center">
    <a href="">Yunheng Wang</a><sup>1,*</sup>
    .
    <a href="">Yuetong Fang</a><sup>1,*</sup>
    .
    <a href="">Taowen Wang</a><sup>1,*</sup>
    .
    <a href="">Lusong Li</a><sup>2</sup>
    .
    <a href="">Kun Liu</a><sup>2</sup>
    .
    <a href="">Junzhe Xu</a><sup>1,2</sup>
    <br/>
    <a href="">Zizhao Yuan</a><sup>1</sup>
    .
    <a href="">Yixiao Feng</a><sup>1</sup>
    .
    <a href="">Jiaxi Zhang</a><sup>1</sup>
    .
    <a href="">Wei Lu</a><sup>1</sup>
    .
    <a href="">Zecui Zeng</a><sup>2,†</sup>
    .
    <a href="">Renjing Xu</a><sup>1,†</sup>
  </p>

  <p align="center"> <strong>Under Review</strong></p>

  <p align="center">
    <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) · 
    <sup>2</sup>JD Explore Academy
  </p>

  <p align="center">
    <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
    <a href=""><img src="https://img.shields.io/badge/🚀_Project_Page-StereoNav-blue" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/YunhengWang/StereoNav_DataSet"><img src="https://img.shields.io/badge/🤗_Dataset-StereoNav_DataSet-yellow" alt="Dataset"></a>
    <a href="https://huggingface.co/YunhengWang/StereoNav_Model"><img src="https://img.shields.io/badge/🤗_Checkpoints-StereoNav_Model-yellow" alt="Checkpoints"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </p>
</p>


# 安装环境

conda create -n stereonav python=3.12.12

conda activate stereonav

pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post2/flash_attn-2.7.1.post2+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install flash_attn-2.7.1.post2+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl


conda env update -f environment.yml



# Evaluation

## Prerequisites: Start Inference Servers

Before running any evaluation, start the model inference servers:

1. Edit `scripts/server.sh`:
   - Set `CHECKPOINT_PATH` to your model checkpoint directory
   - Set `BASE_PORT` (default: 7200) and `NUM_GPUS` (default: 8)

2. Launch servers:
   ```bash
   bash scripts/server.sh
   ```

Each server instance will listen on ports `BASE_PORT` to `BASE_PORT+NUM_GPUS-1`.

---

## 1. Accuracy Evaluation (Exact Goal Points)

Evaluate navigation performance with precise goal-point annotations.

**Configuration:**
- Edit `evaluation/accuracy_eval_p_exact.sh`:
  - Set `DATASET` to `"r2r"` or `"rxr"`
  - Ensure `BASE_PORT` and `NUM_SERVERS` match your running servers

**Run:**
```bash
bash evaluation/accuracy_eval_p_exact.sh
```

**Metrics:** Success Rate, SPL, Oracle Success, Navigation Error, Trajectory Length, nDTW

---

## 2. Accuracy Evaluation (Coarse Goal Points)

Evaluate robustness to goal-point localization errors.

**Configuration:**
- Edit `evaluation/accuracy_eval_p_coarse.sh`:
  - Set `DATASET` to `"r2r"` or `"rxr"`
  - Set `COARSE_RADIUS` (meters) — radius of random sampling around true goal (e.g., `2.0` = ±2m error)
  - Ensure `BASE_PORT` and `NUM_SERVERS` match your running servers

**Run:**
```bash
bash evaluation/accuracy_eval_p_coarse.sh
```

---

## 3. Robustness Evaluation

### 3.1 Lighting Perturbation

Test robustness to lighting changes.

**Configuration:**
- Edit `evaluation/robustness_eval_p_perturbation.sh`:
  - Set `LIGHT_PERTURBATION` — lighting adjustment factor:
    - `0.0` = default lighting
    - Positive values (e.g., `0.8`) = brighter
    - Negative values (e.g., `-0.5`) = darker
    - Range: `-1.0` to `1.0`

**Run:**
```bash
bash evaluation/robustness_eval_p_perturbation.sh
```

---

### 3.2 Camera Height Fluctuation

Test robustness to camera height variations.

**Configuration:**
- Edit `evaluation/robustness_eval_p_fluctuation.sh`:
  - Set `HEIGHT_OFFSET` (meters) — vertical camera displacement:
    - `0.0` = default height (1.25m)
    - Positive values (e.g., `0.6`) = higher camera
    - Negative values (e.g., `-0.6`) = lower camera

**Run:**
```bash
bash evaluation/robustness_eval_p_fluctuation.sh
```

---

### 3.3 Rotation Deviation

Test robustness to viewpoint rotation errors.

**Configuration:**
- Edit `evaluation/robustness_eval_p_fix_deviation.sh`:
  - Set `ROTATION_PERTURBATION` (degrees) — random rotation range per step (e.g., `30` = ±30° deviation)
  - Set `PERTURBATION_PROBABILITY` (0-1) — probability of applying perturbation each step

**Run:**
```bash
bash evaluation/robustness_eval_p_fix_deviation.sh
```

---

### 3.4 Motion Blur

Test robustness to motion blur.

**Configuration:**
- Edit `evaluation/robustness_eval_p_fix_blur.sh`:
  - Set `MOTION_BLUR_STRENGTH` — blur intensity:
    - `0.0` = no blur
    - Higher values = stronger blur (e.g., `1.0`)

**Run:**
```bash
bash evaluation/robustness_eval_p_fix_blur.sh
```

---

## 4. Instruction Ambiguity Analysis (Optional)

Analyze instruction ambiguity using VLM evaluation.

### Step 1: Generate Trajectory Data

Generate ground-truth trajectory frames for analysis:

```bash
bash evaluation/cache/gen_data.sh
```

This creates trajectory data in `evaluation/cache/result_r2r_*` directories.

### Step 2: Run VLM Evaluation

**Configuration:**
- Set your OpenAI API key:
  ```bash
  export OPENAI_API_KEY="your-api-key"
  ```
- Edit `evaluation/pilot_eval_instruction_ambiguity.sh`:
  - Adjust `MODEL` (e.g., `gpt-4o`, `claude-opus-4-7`, `gemini-2.5-pro`)
  - Optionally set `BASE_URL` for custom API endpoints

**Run:**
```bash
bash evaluation/pilot_eval_instruction_ambiguity.sh
```

**Output:** JSONL files with directional and docking ambiguity scores per episode.

---

## Results

All evaluation results are saved to `./results/` by default. Each run generates:
- `result.json` — per-episode metrics and final summary
- Optional: top-down map videos (if `SAVE_VIDEO=true`)


# Training

## Data Preparation

**Option 1: Generate data yourself**
```bash
bash data/gen_trajectory.sh
```
This generates Stage 1 data (R2R, RxR) and Stage 2 data (ScaleVLN).

**Option 2: Download pre-processed data**
```
https://huggingface.co/datasets/YunhengWang/StereoNav_DataSet
```

---

## Stage 1 Training

Default: 8 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu --num_processes 8 --num_machines 1 \
    --mixed_precision bf16 --main_process_port 29555 \
    --dynamo_backend no train_stage1.py
```

---

## Stage 2 Training

### Step 1: Collect DAgger Data

First, start model inference servers:

1. Edit `scripts/server.sh`:
   - Set `CHECKPOINT_PATH` to your Stage 1 checkpoint
   - Set `BASE_PORT` (default: 7200) and `NUM_GPUS` (default: 8)

2. Run:
   ```bash
   bash scripts/server.sh
   ```

Then generate DAgger data for R2R and RxR:

```bash
bash data/dagger.sh
```

Make sure `BASE_PORT` and `NUM_GPUS` in `data/dagger.sh` match your running servers.

### Step 2: Train

Edit `train_stage2.yaml`:
- Set `resume_checkpoint` to your Stage 1 checkpoint path

Run training (default: 8 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --multi_gpu --num_processes 8 --num_machines 1 \
    --mixed_precision bf16 --main_process_port 29555 \
    --dynamo_backend no train_stage2.py
```


# 真机实验


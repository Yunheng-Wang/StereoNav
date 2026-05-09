<p align="center">
  <img src="asset/logo.png" alt="StereoNav Logo" width="120"/>
  <h2 align="center"> What Limits Vision-and-Language Navigation? </h2>
  <p align="center">
    <a href="">Yunheng Wang</a><sup>1</sup>
    ·
    <a href="">Yuetong Fang</a><sup>1</sup>
    ·
    <a href="">Taowen Wang</a><sup>1</sup>
    ·
    <a href="">Lusong Li</a><sup>2</sup>
    ·
    <a href="">Kun Liu</a><sup>2</sup>
    ·
    <a href="">Junzhe Xu</a><sup>1,2</sup>
    <br/>
    <a href="">Zizhao Yuan</a><sup>1</sup>
    ·
    <a href="">Yixiao Feng</a><sup>1</sup>
    ·
    <a href="">Jiaxi Zhang</a><sup>1</sup>
    ·
    <a href="">Wei Lu</a><sup>1</sup>
    <br/>
    <a href="">Zecui Zeng</a><sup>2,†</sup>
    ·
    <a href="">Renjing Xu</a><sup>1,†</sup>
  </p>

  <p align="center"> <strong>Under Review</strong></p>

  <p align="center">
    <sup>1</sup>HKUST(GZ) · 
    <sup>2</sup>JD Explore Academy
  </p>

  <p align="center">
    <img src="asset/HKUSTGZ_logo.svg" alt="HKUST(GZ)" height="40"/>
    &nbsp;&nbsp;&nbsp;
    <img src="asset/JD_logo.png" alt="JD Explore Academy" height="40"/>
  </p>

  <p align="center">
    <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
    <a href=""><img src="https://img.shields.io/badge/🚀_Project_Page-StereoNav-blue" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/YunhengWang/StereoNav_DataSet"><img src="https://img.shields.io/badge/🤗_Dataset-StereoNav_DataSet-yellow" alt="Dataset"></a>
    <a href="https://huggingface.co/YunhengWang/StereoNav_Model"><img src="https://img.shields.io/badge/🤗_Checkpoints-StereoNav_Model-yellow" alt="Checkpoints"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </p>

  <p align="center">
    <img src="asset/main.png" alt="Overview of StereoNav" width="900"/>
  </p>
  <p align="center"><strong>Overview of StereoNav</strong></p>
</p>

## News

- 🔥 **[2026-05-09]** We release all code, model checkpoints, and datasets for StereoNav.

---

## Abstract

Vision-and-Language Navigation (VLN) is a cornerstone of embodied intelligence. However, current agents often suffer from significant performance degradation when transitioning from simulation to real-world deployment, primarily due to perceptual instability (e.g., lighting variations and motion blur) and under-specified instructions. While existing methods attempt to bridge this gap by scaling up model size and training data, we argue that the bottleneck lies in the lack of robust spatial grounding and cross-domain priors. In this paper, we propose StereoNav, a robust Vision-Language-Action framework designed to enhance real-world navigation consistency. To address the inherent gap between synthetic training and physical execution, we introduce Target-Location Priors as a persistent bridge. These priors provide stable visual guidance that remains invariant across domains, effectively grounding the agent even when instructions are vague. Furthermore, to mitigate visual disturbances like motion blur and illumination shifts, StereoNav leverages stereo vision to construct a unified representation of semantics and geometry, enabling precise action prediction through enhanced depth awareness. Extensive experiments on R2R-CE and RxR-CE demonstrate that StereoNav achieves state-of-the-art egocentric RGB performance, with SR and SPL scores of 81.1% and 68.3%, and 67.5% and 52.0%, respectively, while using significantly fewer parameters and less training data than prior scaling-based approaches. More importantly, real-world robotic deployments confirm that StereoNav substantially improves navigation reliability in complex, unstructured environments.

---

# 🛠️ Installation

## Clone the Repository

```bash
git clone https://github.com/Yunheng-Wang/StereoNav.git
cd StereoNav
```

This project requires **two separate conda environments**: one for training and one for evaluation.

---

## Environment 1: Training (`stereonav_train`)

**Step 1:** Create the environment from the provided config:
```bash
conda env create -f environment_train.yml
conda activate stereonav_train
```

**Step 2:** Install PyTorch with the appropriate CUDA version for your system. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct command, e.g.:
```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

**Step 3:** Install `flash-attn`. Download the prebuilt wheel matching your Python, PyTorch, and CUDA versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases), then install it:
```bash
pip install flash_attn-*.whl
```

**Step 4:** Install the matching version of `triton` (should match your PyTorch version):
```bash
pip install triton
```



---

## Environment 2: Evaluation (`stereonav_test`)

**Step 1:** Create the environment and install `habitat-sim`:
```bash
conda create -n stereonav_test python=3.9
conda activate stereonav_test
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
```

**Step 2:** Install `habitat-lab` and `habitat-baselines`:
```bash
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
```

## Data Preparation

### Step 1: Download Scene Data

Download the **MP3D** scenes from [Matterport3D](https://niessner.github.io/Matterport/) and the **HM3D** scenes from [HM3D](https://github.com/matterport/habitat-matterport-3dresearch), then place them under `data/scene/`:

```
data/scene/
├── hm3d/
│   ├── 00000-kfPV7w3FaU5/
│   ├── 00001-UVdNNRcVyV1/
│   ├── 00864-QHhQZWdMpGJ/
│   │   ├── QHhQZWdMpGJ.basis.glb
│   │   └── QHhQZWdMpGJ.basis.navmesh
│   └── ...
└── mp3d/
    ├── 17DRP5sb8fy/
    ├── 1LXtFkjw3qL/
    ├── zsNo4HB9uLZ/
    │   ├── zsNo4HB9uLZ.glb
    │   ├── zsNo4HB9uLZ.house
    │   ├── zsNo4HB9uLZ.navmesh
    │   └── zsNo4HB9uLZ_semantic.ply
    └── ...
```

### Step 2: Download Model Checkpoints

Download the checkpoints from [HuggingFace](https://huggingface.co/anonymous-stereonav/StereoNav_Model) and place them under `checkpoints/`:

| Checkpoint | Description |
|---|---|
| `InternVL3_5-2B` | Base MLLM backbone |
| `dinov2_large` | 2D semantic encoder |
| `edgenext_small` | 2D structural encoder |
| `foundationstereo` | 3D geometry encoder (pre-trained) |
| `stereonav_stage1` | StereoNav Stage 1 checkpoint |
| `stereonav_stage2` | StereoNav Stage 2 checkpoint — ready for evaluation or fine-tuning |

### Step 3: Prepare Training Data

**Option 1: Generate from scratch**
```bash
bash data/gen_trajectory.sh
```
This generates Stage 1 data (R2R, RxR) and Stage 2 data (ScaleVLN).

**Option 2: Download pre-processed data (recommended)**

Download from [HuggingFace](https://huggingface.co/datasets/YunhengWang/StereoNav_DataSet) and place under `data/cache/`:

```
data/cache/
└── train/
    ├── R2R/                    # same structure for RxR and ScaleVLN
    │   ├── images/
    │   ├── annotations_*.json
    │   ├── summary.json
    │   ├── baseline.txt
    │   └── intrinsics.txt
    ├── RxR/
    ├── ScaleVLN/
    └── dagger/
        ├── R2R/
        ├── RxR/
        ├── baseline.txt
        └── intrinsics.txt
```



# 🧪 Evaluation

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


# 🚀 Training

## Data Preparation

Training data can be either generated locally or downloaded from HuggingFace.

**Option 1: Generate from scratch**
```bash
bash data/gen_trajectory.sh
```
This script generates all training data for Stage 1 (R2R, RxR) and Stage 2 (ScaleVLN via DAgger).

**Option 2: Download pre-processed data (recommended)**

Download directly from HuggingFace and place under `data/cache/`:
```
https://huggingface.co/datasets/YunhengWang/StereoNav_DataSet
```

The expected data directory structure is:
```
data/cache/
└── train/
    ├── R2R/
    │   ├── images/
    │   ├── annotations_*.json
    │   ├── summary.json
    │   ├── baseline.txt
    │   └── intrinsics.txt
    ├── RxR/
    │   ├── images/
    │   ├── annotations_*.json
    │   ├── summary.json
    │   ├── baseline.txt
    │   └── intrinsics.txt
    ├── ScaleVLN/
    │   ├── images/
    │   ├── annotations_*.json
    │   ├── summary.json
    │   ├── baseline.txt
    │   └── intrinsics.txt
    └── dagger/
        ├── R2R/
        ├── RxR/
        ├── baseline.txt
        └── intrinsics.txt
```

## Model Checkpoints

Download all required checkpoints from HuggingFace and place under `checkpoints/`:
```
https://huggingface.co/YunhengWang/StereoNav_Model
```

The expected checkpoint directory structure is:
```
checkpoints/
├── InternVL3_5-2B/          # Base MLLM backbone
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json, ...
├── dinov2_large/             # 2D semantic encoder
│   └── model.safetensors
├── edgenext_small/           # 2D structural encoder
│   └── model.safetensors
├── foundationstereo/         # 3D geometry encoder
│   └── ...
├── stereonav_stage1/         # StereoNav Stage 1 checkpoint
│   ├── model.safetensors
│   └── config.json, ...
└── stereonav_stage2/         # StereoNav Stage 2 checkpoint (final)
    ├── model.safetensors
    └── config.json, ...
```



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


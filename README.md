# TM Robot Project: VLA Model Training & Evaluation

A comprehensive robotics project for generating synthetic data, fine-tuning Vision-Language-Action (VLA) models, and evaluating them on pick-and-place tasks using the Techman TM5-700 robot.

## Overview

This repository provides a complete pipeline for:
- **Synthetic data generation** using PyBullet simulation
- **Dataset conversion** to RLDS/TFRecord format
- **Fine-tuning** Vision-Language-Action models (OpenVLA, Pi0)
- **Evaluation** in simulation and on real hardware
- **Teleoperation** tools for data collection and debugging

### Current Performance Metrics
- **Grasp Success Rate (Data Collection)**: ~75%
- **Simulation Environment**: PyBullet
- **Robot**: Techman TM5-700
- **End Effector**: WSG50 Gripper
 
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tm_robot_project.git
cd tm_robot_project
```

### 2. Environment setup
#### Conda
```bash
conda env create -f openvla_nightly_environment.yml
```
#### Docker
Build the container, if you changed the yml, always rebuild the image
```bash
docker build -t tm_robot_vla:latest .
```

To run the container
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v /home/reinaldoyang/openvla_runs:/openvla_runs \
  -e OPENVLA_RUNS_PATH=/openvla_runs \
  tm_robot_vla:latest

```

### 3. Install OpenVLA (for OpenVLA training)
Please refer to OpenVLA official documentation

### 4. Install OpenPi (for Pi0 training)
Please refer to OpenPi official documentation

### 5. Install RLDS Dataset Builder
Please refer to RLDS dataset builder repo

## Quick Start
### Run a Single Simulation

```bash
cd tm_robot_pybullet
python tm_robot_sim.py
```

### Generate Dataset (100 episodes)

```bash
python evaluate_grasp_success.py
```

### Teleoperate the Robot

```bash
# Keyboard control
python teleoperate_tm5.py

# Slider control (GUI)
python tm_robot_slider_test.py
```


## Detailed Usage

### 1. Data Generation
#### Batch Generation
```bash
python automate_data_collection.py --base_dir <location to save dataset> --n_episodes <num of episodes>
```
- Generates any number of dataset you need
- Display successful episodes

#### Check if generated dataset is normal 
```bash
python check_dataset_collection.py
```

### 2. Dataset Conversion

#### Convert JSON to NPY Format

```bash
python convert_json_to_npy.py
```

#### Convert NPY to RLDS/TFRecord

```bash
# Copy NPY data to RLDS builder
cp -r /path/to/source/{train,val} robot_dataset/data/

# Build TFRecord dataset
cd rlds_dataset_builder/robot_dataset
tfds build --overwrite
```

**Output**: `~/tensorflow_datasets/robot_dataset/`

## Model Training

### OpenVLA Fine-tuning

Check out the environment used for training with 5090 gpu

**File**: `openvla_5090.txt`

```bash
cd openvla
conda activate openvla_nightly

torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name robot_dataset \
  --run_root_dir ~/openvla_runs/robot_experiment \
  --adapter_tmp_dir ~/openvla_runs/robot_experiment/adapters \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project robot_finetune \
  --wandb_entity your_wandb_username \
  --save_steps 500 \
  --max_steps 15000
```

### Pi0 Finetuning
1. First convert to lerobot format 
```bash
uv run examples/pybullet/convert_to_lerobot.py --data_dir /path/to/your/data
```
2. Computer normalization
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_bullet_lora_finetune
```

3. Start finetune
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py  pi0_bullet_lora_finetune --exp-name=onepoint --overwrite
```
## Evaluation
### Pi0 Model Evaluation

```bash
cd tm_robot_pybullet
source ~/openpi/.venv/bin/activate  # Use OpenPi environment

python pizero_eval.py
```

### OpenVLA Model Evaluation

```bash
cd openvla
conda activate openvla_nightly

python scripts/evaluate.py \
  --model_path ~/openvla_runs/robot_experiment/checkpoint-15000 \
  --dataset robot_dataset \
  --num_episodes 100
```

**Evaluation Metrics**:
- Success rate across episodes
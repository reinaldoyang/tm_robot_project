# Fine-Tuning OpenVLA on Custom Dataset Generation

This repository was built to generate data and do fine-tuning for Vision-Language-Action (VLA) models.

- Simulation engine: Pybullet

- Robot: Techman TM5-700

- Gripper: WSG50

- Current grasp success rate for data collection : ~75% <br>


# 🚀 Usage
## Simulation environment using PyBullet for Techman Robot TM5-700 robot

run this to get a simulation once
```bash
python tm_robot_sim.py
```

to generate dataset, run this, this will also output the number of successful dataset generated
```bash
python evaluate_grasp_success.py
```

to control the robot using slider, run this command
```bash
python tm_robot_slider_test.py
```

to convert the generated dataset(img and json file) to npy file format
```bash
python convert_json_to_npy.py
```

after converting the original format dataset to npy, copy the data file which contain train and val to rlds_dataset_buider/robot_dataset data folder
use rlds_env in order to convert the npy file into tfds format dataset using this command below:
```bash
cd robot_dataset
tfds build --overwrite
```



## Real Robot
```bash
cd tm_robot_real
```



cd openvla folder
conda activate openvla_nightly
run this finetuning script





```bash
torchrun --standalone --nnodes 1 
--nproc-per-node 1 vla-scripts/finetune.py 
--vla_path "openvla/openvla-7b" 
--data_root_dir ~/tensorflow_datasets 
--dataset_name robot_dataset 
--run_root_dir ~/openvla_runs/robot_experiment_4 
--adapter_tmp_dir ~/openvla_runs/robot_experiment_4/adapters 
--lora_rank 32 
--batch_size 4 
--grad_accumulation_steps 8 
--learning_rate 5e-4 
--image_aug True 
--wandb_project robot_finetune 
--wandb_entity reinaldoyang5-national-cheng-kung-university-co-op 
--save_steps 500 
--max_steps 15000
```

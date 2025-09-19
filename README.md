# Fine-Tuning OpenVLA on Custom Dataset Generation

This repository was built to generate data and do fine-tuning for Vision-Language-Action (VLA) models.

- Simulation engine: Pybullet

- Robot: Techman TM5-700

- Gripper: WSG50

- Current grasp success rate for data collection : ~75% <br>


# 🚀 Usage
Simulation environment using PyBullet

run this to get a simulation once
```bash
python tm_robot_sim.py
```

to generate dataset, run this
```bash
python evaluate_grasp_success.py
```

to control the robot using slider, run this
```bash
python tm_robot_slider_test.py
```

to convert the generated rlds dataset to npy file format
```bash
python convert_json_to_npy.py
```

after converting the generated rlds format dataset to npy, copy the train and val data to rlds_dataset_buider/robot_dataset data folder
use rlds_env in order to convert the npy file into tfds format dataset using this command below:
```bash
tfds build --overwrite
```
cd openvla folder
conda activate openvla_nightly
run this finetuning script
```bash
# Common used commands

Edit path file:

```bash
gedit ~/.bashrc
```

To download the nvidia 570 open:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-570-open
```

To run docker for tensorflow, to solve the 5000 blackwell gpu not working

```bash
docker run --gpus all -it -v /home/reinaldoyang/tm_robot_project:/workspace nvcr.io/nvidia/tensorflow:25.02-tf2-py3

```

`-v /home/reinaldoyang/rlds_dataset_builder:/workspace` to mount the folder from host to container

docker cheat sheet

docker ps -a

fine tuning command 

to start the container that are already created once

```bash

docker start -ai tm_robot_container

```

run container with dataset mounted

```bash

docker run --gpus all -it \

--name tm_robot_container \

-v /home/reinaldoyang/tm_robot_project/rlds_dataset_builder/rlds_dataset_npy:/workspace/rlds_dataset_npy \

tm_robot_env

```


use pytorch 2.80, openvla_nightly conda env

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path "openvla/openvla-7b" --data_root_dir ~/tensorflow_datasets --dataset_name robot_dataset --run_root_dir ~/openvla_runs/robot_experiment_4 --adapter_tmp_dir ~/openvla_runs/robot_experiment_4/adapters --lora_rank 32 --batch_size 4 --grad_accumulation_steps 8 --learning_rate 5e-4 --image_aug True --wandb_project robot_finetune --wandb_entity reinaldoyang5-national-cheng-kung-university-co-op --save_steps 500 --max_steps 15000

<<<<<<< HEAD
```
=======
```

>>>>>>> 70bd5039493983ce109d1e9c4810df7f0a2dc4f7

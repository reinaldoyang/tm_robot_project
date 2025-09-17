# Fine-Tuning OpenVLA on Custom Dataset Generation

This repository was built to generate data and do fine-tuning for Vision-Language-Action (VLA) models.

- Simulation engine: Pybullet

- Robot: Techman TM5-700

- Gripper: WSG50

- Current grasp success rate for data collection : ~75%

🚀 Usage
Simulation environment using PyBullet

Robot used in this project: Techman robot TM5-700

Gripper used in this project: WSG50

run conda myenv
tm_robot_sim for techman robot tm5-700 simulation using pybullet 

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

after converting the generated rlds format dataset to npy, copy the train and val data to rlds_dataset_buider/robot_dataset folder

and then build to tensorflow dataset format TFDS



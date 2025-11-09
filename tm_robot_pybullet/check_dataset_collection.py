from tm_robot_sim import create_simulation_env, attach_gripper_to_robot
import json
import time
from finetuned_evaluation import apply_action_to_robot
import pybullet as p
import os

def replay_episode(json_path):
    #load dataset episode
    with open(json_path, "r") as f:
        episode = json.load(f)

    ee_states = episode["ee_states"]

    plane_id, robot_id, table_id, cube_id, tray_id, gripper_id = create_simulation_env("GUI")
    attach_gripper_to_robot(robot_id, gripper_id)
    end_effector_idx = 6
    arm_joints = [1, 2, 3, 4, 5, 6]

    # Simulation / control parameters
    control_dt = 0.2      # 5 Hz control
    physics_dt = 1/240    # physics runs at 240 Hz
    steps_per_control = int(control_dt / physics_dt)

    for t, ee in enumerate(ee_states):
        print(f"Step {t+1}/{len(ee_states)}: target EE = {ee}")

        apply_action_to_robot(robot_id, gripper_id, ee, arm_joints)

        # Step physics for control_dt duration
        for _ in range(steps_per_control):
            p.stepSimulation()
            time.sleep(physics_dt)  # keep real-time sync'
    p.disconnect()
    print(f"Finished replaying episode {episode['episode_id']}")


def replay_all_episodes(dataset_dir):
    for ep_name in sorted(os.listdir(dataset_dir)):
        ep_path = os.path.join(dataset_dir, ep_name, f"{ep_name}.json")
        if os.path.isfile(ep_path):
            replay_episode(ep_path)

if __name__ == "__main__":
    dataset_dir = "rlds_onep_spawn"
    replay_all_episodes(dataset_dir)

import sys
sys.path.append('/home/reinaldoyang/openpi/src')
from openpi.training import config as _config
from openpi.policies import policy_config
from tm_robot_sim import create_simulation_env, attach_gripper_to_robot, capture_image, check_task_success, get_end_effector_state
import pybullet as p
import numpy as np
import time
import collections 
import os
import json
import random
from finetuned_evaluation import save_video

def save_gripper_states(gripper_states, episode_idx, save_dir="gripper state"):
    """
    Save the list of gripper states to a JSON file for a given episode.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"gripper_states_ep{episode_idx:03d}.json")
    with open(save_path, "w") as f:
        json.dump(gripper_states, f)
    print(f"Gripper states saved to {save_path}")

def load_pi0_policy():
    config = _config.get_config("pi05_bullet")
    checkpoint_dir = "/home/reinaldoyang/openpi/checkpoints/pi05_bullet/pi05_50hz_ah50/9999"
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    return policy

def predict_action(policy, image, robot_state, prompt):
    example = {
        "observation/image":image,
        "observation/state": robot_state,    
        "prompt": prompt
    }
    action_chunk = policy.infer(example)["actions"]
    return action_chunk

def apply_action_to_robot(robot_id, gripper_id, action, arm_joints):
    if hasattr(action, "numpy"):
        action = action.numpy()
    elif not isinstance(action, np.ndarray):
        action = np.array(action)
    
    action = action.flatten()
    
    pos = action[:3].tolist()
    orn = p.getQuaternionFromEuler(action[3:6].tolist())
    gripper_val = float(action[6])
    joint_positions = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=6,
        targetPosition=pos,
        targetOrientation=orn
    )
    for i, j in enumerate(arm_joints):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_positions[i],
            force=200,
            maxVelocity=3
        )
    p.setJointMotorControl2(gripper_id, 4, p.POSITION_CONTROL,
                            targetPosition=gripper_val * 0.05, force=100)
    p.setJointMotorControl2(gripper_id, 6, p.POSITION_CONTROL,
                            targetPosition=gripper_val * 0.05, force=100)


def run_real_time_pi0(policy, replan_steps=1, episode_idx = 0, save_dir="gripper state", save_video_dir="videos"):
    plane_id, robot_id, table_id, cube_id, tray_id, gripper_id = create_simulation_env("others")
    attach_gripper_to_robot(robot_id, gripper_id)
    arm_joints = [1,2,3,4,5,6]
    end_effector_idx = 6
    cam_width, cam_height = 224, 224
    prompt = "put white cube on tray"
    control_dt = 0.02
    physics_dt = 1/240
    steps_per_control = int(control_dt / physics_dt)
    
    # Action queue for storing predicted action chunks
    action_plan = collections.deque()
    gripper_states = []
    frames = []
    
    try:
        step_count = 0
        gripper_state = 0
        while step_count < 1200:
            p.stepSimulation()
            time.sleep(physics_dt)
            step_count += 1
            
            if step_count % steps_per_control == 0:
                # Check if we need to query the model for a new action chunk
                if not action_plan:
                    # Action queue is empty - get new predictions
                    image = capture_image(cam_width, cam_height)
                    ee_pos, ee_rpy= get_end_effector_state(robot_id, end_effector_idx)
                    robot_state = np.array([*ee_pos, *ee_rpy, gripper_state])
                    
                    # Get action chunk from policy
                    action_chunk = predict_action(policy, image, robot_state, prompt)
                    # print(f"Predicted action chunk shape: {action_chunk.shape}")
                    
                    # Add actions to the queue (only use replan_steps actions)
                    for i in range(min(replan_steps, len(action_chunk))):
                        action_plan.append(action_chunk[i])
                
                # Pop one action from the queue and execute it
                action = action_plan.popleft()
                apply_action_to_robot(robot_id, gripper_id, action, arm_joints)
                new_gripper_state = float(action[6])
                gripper_state = np.clip(new_gripper_state, 0.0, 1.0)
                gripper_states.append(gripper_state)
                if step_count % 8 == 0:
                    image = capture_image(cam_width, cam_height)
                    frames.append(image)
        
        save_gripper_states(gripper_states, episode_idx, save_dir)
        
        cube_now = p.getBasePositionAndOrientation(cube_id)[0]
        tray_pos = p.getBasePositionAndOrientation(tray_id)[0]
        task_success = check_task_success(cube_now, tray_pos)
        print("✅ Task success:", task_success)
        return task_success, frames
    finally:
        p.disconnect()

def evaluate_pi0_model(num_episodes = 300, save_video_dir="videos"):
    policy = load_pi0_policy()
    successes = 0
    unsuccessful_episodes = []  # Store (episode_idx, frames) for unsuccessful episodes
    cam_width, cam_height = 224, 224
    
    for ep in range(num_episodes):
        print(f"=== Running episode {ep+1}/{num_episodes} ===")
        success, frames = run_real_time_pi0(policy, replan_steps=5, episode_idx=ep)
        
        if success:
            successes += 1
            print("✅ Success")
            # Save video for all successful episodes
            save_video(frames, save_video_dir, f"{ep:03d}_success", cam_width, cam_height)
        else:
            print("❌ Fail")
            # Store unsuccessful episode data for potential random selection
            unsuccessful_episodes.append((ep, frames))
    
    # Randomly select 3 unsuccessful episodes to save videos
    if len(unsuccessful_episodes) > 0:
        num_to_save = min(3, len(unsuccessful_episodes))
        selected_unsuccessful = random.sample(unsuccessful_episodes, num_to_save)
        
        print(f"\n📹 Saving videos for {num_to_save} randomly selected unsuccessful episodes:")
        for ep_idx, frames in selected_unsuccessful:
            save_video(frames, save_video_dir, f"{ep_idx:03d}_fail", cam_width, cam_height)
            print(f"   - Episode {ep_idx}")
    
    success_rate = successes / num_episodes
    print(f"\n📊 Evaluation finished: {successes}/{num_episodes} successful episodes")
    print(f"➡️ Success rate: {success_rate*100:.2f}%")
    print(f"🎥 Videos saved: {successes} successful + {min(3, len(unsuccessful_episodes))} unsuccessful")
    return success_rate
    

if __name__ == "__main__":
    evaluate_pi0_model(num_episodes=50)




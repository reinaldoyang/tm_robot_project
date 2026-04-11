from tm_robot_sim import create_simulation_env, attach_gripper_to_robot, capture_image, check_grasp_success, check_task_success
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pybullet as p
import time
import torch
import math
import numpy as np
import os
import json
import cv2 as cv

def load_vla_model(model_name, dataset_stats_path = None, device = "cuda:0"):
    """
    load openvla model
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code = True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
        trust_remote_code = True
    ).to(device)
    vla.eval()

    #load dataset statistics from fine-tuning
    unnorm_key = None
    if dataset_stats_path is not None:
        import json
        with open(dataset_stats_path, "r") as f:
            vla.norm_stats = json.load(f)
            unnorm_key = list(vla.norm_stats.keys())[0]

    return processor, vla, unnorm_key

def predict_action(vla, processor, prompt, image, unnorm_key, device="cuda:0"):
    """
    predict next robot action given camera image and prompt
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply center crop augmentation (same as training)
    crop_scale = 0.9
    sqrt_crop_scale = math.sqrt(crop_scale)
    
    temp_image = np.array(image)  # Convert to numpy (H, W, C)
    temp_image_cropped = apply_center_crop(
        temp_image, 
        t_h=int(sqrt_crop_scale * temp_image.shape[0]), 
        t_w=int(sqrt_crop_scale * temp_image.shape[1])
    )
    temp_image = Image.fromarray(temp_image_cropped)
    temp_image = temp_image.resize(image.size, Image.Resampling.BILINEAR)  # IMPORTANT: BILINEAR resize
    image = temp_image
    
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action

def apply_center_crop(image, t_h, t_w):
    """
    Apply center crop to image
    Args:
        image: numpy array (H, W, C)
        t_h: target height
        t_w: target width
    Returns:
        cropped image (t_h, t_w, C)
    """
    h, w = image.shape[:2]
    top = (h - t_h) // 2
    left = (w - t_w) // 2
    return image[top:top+t_h, left:left+t_w]

def apply_action_to_robot(robot_id, gripper_id, action, arm_joints):
    #split action 
    pos = action[:3]
    orn = p.getQuaternionFromEuler(action[3:6]) 
    gripper_val = action[6]
    joint_positions = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=6,
        targetPosition=pos,
        targetOrientation=orn
    )

    # Apply joint controls
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

def set_initial_robot_pose(robot_id, arm_joints, end_effector_idx, start_pos):
    """
    Set the robot's initial pose to a specific position with roll=0, pitch=180, yaw=90
    """
    # Desired orientation
    target_orn = p.getQuaternionFromEuler([0, math.pi, math.pi/2])

    # Compute IK for the specified start position
    joint_poses = p.calculateInverseKinematics(robot_id, end_effector_idx, start_pos, target_orn)

    # Apply joint positions
    for i, j in enumerate(arm_joints):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_poses[i],
            force=200,
            maxVelocity=3
        )

def save_video(frames, save_video_dir, episode_id, cam_width, cam_height, playback_fps=10):
    """
    Save the collected frames as a video.
    """
    if not os.path.exists(save_video_dir):
        os.makedirs(save_video_dir)
    video_path = os.path.join(save_video_dir, f"episode_{episode_id}.mp4")

    # Initialize video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv.VideoWriter(video_path, fourcc, playback_fps, (cam_width, cam_height))

    # Write frames to video
    for frame in frames:
        frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        video_writer.write(frame_bgr)

    video_writer.release()  # Release the video writer
    print(f"📁 Video saved: {video_path}")

def run_real_time(vla, processor, unnorm_key, device="cuda:0", save_video_dir="evaluation_vid", episode_idx = 0):
    plane_id, robot_id, table_id, cube_id, tray_id, gripper_id = create_simulation_env("others")
    attach_gripper_to_robot(robot_id, gripper_id)
    arm_joints = [1, 2, 3, 4, 5, 6]
    cam_width, cam_height = 224, 224
    prompt = "In: What action should the robot take to put cube in tray?\nOut:"
    control_dt = 0.2 #update control every 0.2s
    physics_dt = 1 / 240 #240hz
    steps_per_control = int(control_dt / physics_dt) #do 48 physics step per control step

    # Initialize video writer
    frames = []

    try:
        step_count = 0
        while step_count < 450:
            # Step the simulation
            p.stepSimulation() #physic step
            time.sleep(physics_dt)
            step_count += 1 #how many physic step have passed

            if step_count % steps_per_control == 0: #run every 48 physics steps
                # Capture image from the camera
                image = capture_image(cam_width, cam_height)

                # Predict action and apply it to the robot
                action = predict_action(vla, processor, prompt, image, unnorm_key, device)
                apply_action_to_robot(robot_id, gripper_id, action, arm_joints)

                # Collect frames for video every 8th step, comment if don't want to save vide
                if step_count % 8 == 0:
                    frames.append(image)

        # Check task success
        cube_now = p.getBasePositionAndOrientation(cube_id)[0]
        tray_pos = p.getBasePositionAndOrientation(tray_id)[0]
        task_success = check_task_success(cube_now, tray_pos)
        print("✅ Task success:" if task_success else "❌ Task failed:", task_success)

        # Save video for all evaluations
        result_dir = "successful" if task_success else "unsuccessful"
        episode_dir = os.path.join(save_video_dir, result_dir)
        os.makedirs(episode_dir, exist_ok=True)
        episode_id = episode_idx
        save_video(frames, episode_dir, episode_id, cam_width, cam_height, playback_fps=5)

        return task_success
    finally:
        p.disconnect()

def evaluate_model(num_episodes = 100):
    open_vla_weights_path = '/home/reinaldoyang/openvla_runs/robot_experiment/openvla-7b+robot_dataset+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug'
    dataset_stats_path = "/home/reinaldoyang/openvla_runs/robot_experiment/openvla-7b+robot_dataset+b32+lr-0.0005+lora-r32+dropout-0.0--image_aug/dataset_statistics.json"
    device = "cuda:0"

    processor, vla, unnorm_key = load_vla_model(open_vla_weights_path, 
                                    dataset_stats_path,
                                    device=device)
    successes = 0
    for ep in range(num_episodes):
        print(f"=== Running episode {ep+1}/{num_episodes} ===")
        success = run_real_time(vla, processor, unnorm_key, device, episode_idx = ep)
        if success:
            successes += 1
            print("✅ Success")
        else:
            print("❌ Fail")

    success_rate = successes / num_episodes
    print(f"\n📊 Evaluation finished: {successes}/{num_episodes} successful episodes")
    print(f"➡️ Success rate: {success_rate*100:.2f}%")

    return success_rate


if __name__ == "__main__":
    evaluate_model(num_episodes=100)

import sys
sys.path.append('/home/reinaldoyang/openpi/src')
from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np
import json
import os
from PIL import Image

def load_pi0_policy():
    """Load the trained policy"""
    config = _config.get_config("pi0_bullet_lora_finetune")
    checkpoint_dir = "/home/reinaldoyang/openpi/checkpoints/pi0_bullet_lora_finetune/onepoint_5_hor/4999"
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    return policy

def predict_action(policy, image, robot_state, prompt):
    """Same prediction function as in your eval script"""
    example = {
        "observation/image": image,
        "observation/state": robot_state,
        "prompt": prompt
    }
    action_chunk = policy.infer(example)["actions"]
    return action_chunk

def load_training_episode(episode_dir):
    """Load a training episode where actions are the next ee_states"""
    episode_name = os.path.basename(episode_dir)
    json_path = os.path.join(episode_dir, f"{episode_name}.json")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Load images from the img subfolder
    images = []
    img_folder = os.path.join(episode_dir, "img")
    
    for img_name in data['img_filenames']:
        img_path = os.path.join(img_folder, img_name)
        img = Image.open(img_path)
        images.append(np.array(img))
    
    ee_states = data['ee_states']
    
    # Create actions as the next ee_state
    # For step i, action is ee_states[i+1]
    # So we have len(ee_states)-1 action-state pairs
    current_states = ee_states[:-1]  # All states except the last one
    actions = ee_states[1:]          # All states except the first one (these are the targets)
    images = images[:-1]             # Remove last image to match

    return images, current_states, actions

def evaluate_on_training_data(rlds_dir="rlds_1p_spawn_opt", max_print_steps=1, max_print_episodes=2):
    """Evaluate model performance on training data.

    This evaluates ALL episodes found in `rlds_dir` for the overall averages,
    but only prints detailed output for up to `max_print_episodes` episodes
    and up to `max_print_steps` steps per printed episode.
    """
    policy = load_pi0_policy()

    # Evaluate ALL episodes in the folder for averages
    episode_dirs = sorted([d for d in os.listdir(rlds_dir) if d.startswith("episode_")])
    
    total_steps = 0
    total_position_error = 0
    total_orientation_error = 0
    total_gripper_error = 0
    
    prompt = "pick up the white cube"  # Match your training prompt
    
    print(f"Evaluating model on {len(episode_dirs)} training episodes...")
    print("="*80)

    printed_episodes = 0
    
    for ep_dir in episode_dirs:
        episode_path = os.path.join(rlds_dir, ep_dir)
        # Decide whether to print detailed info for this episode
        print_this_episode = printed_episodes < max_print_episodes
        if print_this_episode:
            print(f"\nEpisode: {ep_dir}")
        
        # Load training data (actions are next ee_states)
        images, states, gt_actions = load_training_episode(episode_path)
        
        episode_pos_error = 0
        episode_orn_error = 0
        episode_grip_error = 0
        
        # Print step count only when printing this episode's details
        if print_this_episode:
            print(f"  Steps: {len(images)} (note: {len(images)} action-state pairs from {len(images)+1} total states)")
        
        for step in range(len(images)):
            # Get ground truth
            gt_state = np.array(states[step])
            gt_next_state = np.array(gt_actions[step])  # This is the next ee_state (target)
            
            # Predict action
            try:
                predicted_chunk = predict_action(policy, images[step], gt_state, prompt)
                pred_action = predicted_chunk[0]  # Take first action from chunk
                
                # Calculate errors (comparing predicted action to ground truth next state)
                pos_error = np.linalg.norm(gt_next_state[:3] - pred_action[:3])  # Position error
                orn_error = np.linalg.norm(gt_next_state[3:6] - pred_action[3:6])  # Orientation error  
                grip_error = abs(gt_next_state[6] - pred_action[6])  # Gripper error
                
                episode_pos_error += pos_error
                episode_orn_error += orn_error
                episode_grip_error += grip_error
                
                # Only print detailed per-step info for the first `max_print_episodes` episodes
                # and limit per-episode prints to `max_print_steps` steps.
                if print_this_episode and step < max_print_steps:
                    print(f"    Step {step}:")
                    print(f"      Current:   [{gt_state[0]:.3f}, {gt_state[1]:.3f}, {gt_state[2]:.3f}, {gt_state[3]:.3f}, {gt_state[4]:.3f}, {gt_state[5]:.3f}, {gt_state[6]:.3f}]")
                    print(f"      GT next:   [{gt_next_state[0]:.3f}, {gt_next_state[1]:.3f}, {gt_next_state[2]:.3f}, {gt_next_state[3]:.3f}, {gt_next_state[4]:.3f}, {gt_next_state[5]:.3f}, {gt_next_state[6]:.3f}]")
                    print(f"      Predicted: [{pred_action[0]:.3f}, {pred_action[1]:.3f}, {pred_action[2]:.3f}, {pred_action[3]:.3f}, {pred_action[4]:.3f}, {pred_action[5]:.3f}, {pred_action[6]:.3f}]")
                    print(f"      Errors: pos={pos_error:.4f}, orn={orn_error:.4f}, grip={grip_error:.4f}")
                
            except Exception as e:
                print(f"    Error at step {step}: {e}")
                continue
        
        # Episode averages
        num_steps = len(images)
        avg_pos_error = episode_pos_error / num_steps
        avg_orn_error = episode_orn_error / num_steps  
        avg_grip_error = episode_grip_error / num_steps
        
        # Print episode-level averages only for the printed episodes to reduce output noise
        if print_this_episode:
            print(f"  Avg Position Error: {avg_pos_error:.4f}")
            print(f"  Avg Orientation Error: {avg_orn_error:.4f}")
            print(f"  Avg Gripper Error: {avg_grip_error:.4f}")
            # mark that we've printed one more episode
            printed_episodes += 1
        
        # Accumulate totals
        total_steps += num_steps
        total_position_error += episode_pos_error
        total_orientation_error += episode_orn_error
        total_gripper_error += episode_grip_error
    
    # Overall averages
    overall_pos_error = total_position_error / total_steps
    overall_orn_error = total_orientation_error / total_steps
    overall_grip_error = total_gripper_error / total_steps
    
    print("\n" + "="*80)
    print("OVERALL RESULTS:")
    print(f"Total steps evaluated: {total_steps}")
    print(f"Average Position Error: {overall_pos_error:.4f} meters")
    print(f"Average Orientation Error: {overall_orn_error:.4f} radians")
    print(f"Average Gripper Error: {overall_grip_error:.4f}")
    print("="*80)
    
    return {
        'position_error': overall_pos_error,
        'orientation_error': overall_orn_error, 
        'gripper_error': overall_grip_error,
        'total_steps': total_steps
    }

if __name__ == "__main__":
    # Test on a few training episodes
    results = evaluate_on_training_data(max_print_steps=20, max_print_episodes=2)
    
    print("\nInterpretation:")
    print(f"Position error of {results['position_error']:.4f}m = {results['position_error']*100:.1f}cm")
    print(f"Orientation error of {results['orientation_error']:.4f} rad = {np.degrees(results['orientation_error']):.1f} degrees")
    print(f"Gripper error of {results['gripper_error']:.4f} (0=perfect, 1=completely wrong)")
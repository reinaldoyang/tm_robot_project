import numpy as np
import argparse
from tm_robot_sim import run_simulation, save_rlds_episode

def run_multiple_episodes(n_episodes=100, base_dir=None):
    if base_dir is None:
        raise ValueError("Please specify a base directory to save episodes")
    grasp_success_list = []
    episode_counter = 1

    for i in range(n_episodes):
        print(f"Running episode {i+1}/{n_episodes}...")
        grasp_success, task_success, frames, ee_states, cube_pos, tray_pos = run_simulation() 
        grasp_success_list.append(grasp_success and task_success)

        if grasp_success and task_success:
            episode_id = f"episode_{episode_counter:03d}"
            save_rlds_episode(base_dir, episode_id, frames, ee_states, cube_pos, tray_pos)
            print(f"Saved RLDS episode: {episode_id}")
            episode_counter += 1

    grasp_success_array = np.array(grasp_success_list)  
    success_rate = np.sum(grasp_success_array) / n_episodes
    print(f"\nGrasp + task success rate over {n_episodes} episodes: {success_rate*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate grasp success and save episodes")
    parser.add_argument("--base_dir", type=str, required=True, help ="Directory to save episodes")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of episodes to run")
    args = parser.parse_args()

    run_multiple_episodes(n_episodes=args.n_episodes, base_dir=args.base_dir)

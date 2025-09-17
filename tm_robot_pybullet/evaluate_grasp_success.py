import numpy as np
from tm_robot_sim import run_simulation, save_rlds_episode



def run_multiple_episodes(n_episodes=300, base_dir = "rlds_dataset_2"):
    grasp_success_list = []
    episode_counter = 1

    for i in range(n_episodes):
        print(f"Running episode {i+1}/{n_episodes}...")
        # Run simulation in DIRECT mode to speed it up
        grasp_success, task_success, frames, ee_states = run_simulation() 
        grasp_success_list.append(grasp_success and task_success)

        if grasp_success and task_success:
            episode_id = f"episode_{episode_counter:03d}"
            save_rlds_episode(base_dir, episode_id, frames, ee_states)
            print(f"Saved RLDS episode: {episode_id}")
            episode_counter += 1

    grasp_success_array = np.array(grasp_success_list)  
    success_rate = np.sum(grasp_success_array) / n_episodes
    print(f"\nGrasp + task success rate over {n_episodes} episodes: {success_rate*100:.2f}%")

if __name__ == "__main__":
    run_multiple_episodes()

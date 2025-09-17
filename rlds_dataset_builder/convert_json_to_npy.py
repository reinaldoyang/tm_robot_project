import os
import json
import numpy as np
from PIL import Image
import shutil
import random

def convert_json_to_npy(json_path, img_base_path, save_path):
    """
    Convert a single episode from json format to npy format
    """
    #check if dir exist, if not then create one
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    timesteps = data['timesteps']
    img_filenames = data['img_filenames']
    ee_states = data['ee_states']

    episode = []

    for i in range(timesteps):
        #Convert img into np array
        img_path = os.path.join(img_base_path, img_filenames[i])
        img_array = np.array(Image.open(img_path))
        
        if i < timesteps - 1: #break if current i equal the last element
            action = np.array(ee_states[i+1], dtype = np.float32)
        else:
            action = np.zeros_like(ee_states[i], dtype = np.float32)

        step_dict = {
            'image': img_array,
            'state' : np.array(ee_states[i], dtype = np.float32),
            'action': action,
            'language_instruction': "pick up the white cube"
        }
        episode.append(step_dict)

    np.save(save_path, episode, allow_pickle = True)
    print(f"Saved {save_path} with {timesteps} steps")

def convert_all_eps(rlds_dataset_dir, save_dir):
    os.makedirs(save_dir, exist_ok = True) #create save directory
    all_items = os.listdir(rlds_dataset_dir) #get all of the items in this dataset
    episode_folders = []
    for items in all_items:
        complete_path = os.path.join(rlds_dataset_dir, items) #combine the main folder and the folder name
        if os.path.isdir(complete_path): #check if true
            episode_folders.append(items)

    #process each episode in the episode_folders
    for episode_folder in episode_folders:
        episode_path = os.path.join(rlds_dataset_dir, episode_folder)
        json_files = []
        for f in os.listdir(episode_path):
            if f.endswith(".json"):
                json_files.append(f)
        if len(json_files) == 0:
            continue #skip if no json file found
        #every file have only one json
        json_path = os.path.join(episode_path, json_files[0])
        img_base_path = os.path.join(episode_path, "img")
        save_path = os.path.join(save_dir, f"{episode_folder}.npy")
        convert_json_to_npy(json_path, img_base_path, save_path)

def split_dataset():
    # Paths
    dataset_dir = "rlds_dataset_npy"      # your current folder with all .npy episodes
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # Create folders if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all .npy files
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".npy")]

    # Shuffle randomly
    random.shuffle(all_files)

    # Split index
    split_idx = int(len(all_files) * 0.8)  # 80% train

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Move files to respective folders
    for f in train_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(train_dir, f))

    for f in val_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(val_dir, f))

    print(f"Total episodes: {len(all_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

if __name__ == "__main__":
    rlds_dataset_dir = "../tm_robot_pybullet/rlds_dataset_2"
    save_dir = "rlds_dataset_npy"
    convert_all_eps(rlds_dataset_dir, save_dir)
    split_dataset()
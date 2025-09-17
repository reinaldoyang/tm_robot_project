import numpy as np

# Load a single episode
episode_path = "robot_dataset/data/train/episode_014.npy"
data = np.load(episode_path, allow_pickle=True)  # data is a list of dicts

print("Type:", type(data))
print("Length:", len(data))
print("First step keys:", data[0].keys())
print("Example state shape:", data[0]["state"].shape)
print("Example action shape:", data[0]["action"].shape)
print("Language instruction:", data[0]["language_instruction"])
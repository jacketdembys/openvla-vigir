import os
import numpy as np
import json
from glob import glob
from PIL import Image
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R

# Global dictionary for language instructions per episode folder
LANGUAGE_INSTRUCTIONS = {
    # Example entries; user should fill these out as needed
    '3_cups': ['Stack the three cups'],
    'cup_on_cup_4': ['Stack the cups'],
    'cup_on_marker': ['Put the cup on marker'],
    'marker_on_cup': ['Put the marker in the cup'],
    'yellow_on_cup': ['Put the yellow object on the cup']
}

def load_image_as_np(path):
    return np.array(Image.open(path))

def build_state(entry):
    pos = entry['position']
    orientation = entry['orientation']
    fingers = entry['fingers']
    # Use scipy for quaternion to roll, pitch, yaw
    quat = [orientation['x'], orientation['y'], orientation['z'], orientation['w']]
    roll, pitch, yaw = R.from_quat(quat).as_euler('xyz', degrees=False)
    finger_val = 1.0 if any(f > 3500 for f in [fingers['finger1'], fingers['finger2'], fingers['finger3']]) else 0.0
    return np.array([
        pos['x'], pos['y'], pos['z'],
        roll, pitch, yaw,
        finger_val
    ], dtype=np.float32)

def build_action(entry):
    dp = entry['delta_position']
    dor = entry['delta_orientation_rpy']
    df = entry['delta_fingers']
    # Use 1.0 if any delta finger above 3500, else 0.0
    finger_val = 1.0 if any(f > 3500 for f in [df['finger1'], df['finger2'], df['finger3']]) else 0.0
    return np.array([
        dp['x'], dp['y'], dp['z'],
        dor['roll'], dor['pitch'], dor['yaw'],
        finger_val
    ], dtype=np.float32)

def create_camera_episodes(episode_dir, json_path, save_dir, episode_name=None):
    # Use the global LANGUAGE_INSTRUCTIONS dict and fallback to default if not present
    if episode_name is None:
        episode_name = os.path.basename(episode_dir)
    language_instructions = LANGUAGE_INSTRUCTIONS.get(episode_name, [f'instruction {i+1}' for i in range(5)])
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = sorted(data, key=lambda x: x['frame'])
    # Get camera serials from first frame folder
    first_frame_folder = os.path.join(episode_dir, str(data[0]['frame']))
    rgb_files = glob(os.path.join(first_frame_folder, '*_rgb.jpg'))
    camera_serials = [os.path.basename(f).split('_')[1] for f in rgb_files]
    print(f'Camera serials: {camera_serials}')
    # Create an episode for each camera, for each instruction
    os.makedirs(save_dir, exist_ok=True)
    episode_dict = {}
    for instruction in language_instructions:
        instruction_episodes = {}
        for serial in camera_serials:
            episode = []
            for entry in tqdm(data):
                frame_num = entry['frame']
                frame_folder = os.path.join(episode_dir, str(frame_num))
                rgb_path = os.path.join(frame_folder, f'camera_{serial}_rgb.jpg')
                depth_path = os.path.join(frame_folder, f'camera_{serial}_depth.png')
                image = load_image_as_np(rgb_path) if os.path.exists(rgb_path) else None
                depth_image = load_image_as_np(depth_path) if os.path.exists(depth_path) else None
                state = build_state(entry)
                action = build_action(entry)
                episode.append({
                    'image': image,
                    'depth_image': depth_image,
                    'state': state,
                    'action': action,
                    'language_instruction': instruction,
                })
            save_path = os.path.join(save_dir, f'episode_fake_camera_{serial}_instr_{instruction.replace(" ", "_")}.npy')
            print(f'Saving camera {serial} episode with instruction "{instruction}" to {save_path}')
            np.save(save_path, episode)
            instruction_episodes[serial] = save_path
        episode_dict[instruction] = instruction_episodes
    # Save the dict as a json for reference
    dict_save_path = os.path.join(save_dir, 'episode_index.json')
    with open(dict_save_path, 'w') as f:
        json.dump(episode_dict, f, indent=2)
    print(f'Saved episode index dict to {dict_save_path}')

def main():
    root = '/home/Desktop/ViGIR_VLA_Dataset'
    save_root = '/home/Desktop/Saved_Episodes'
    os.makedirs(save_root, exist_ok=True)
    for episode_name in os.listdir(root):
        episode_dir = os.path.join(root, episode_name)
        if not os.path.isdir(episode_dir):
            continue
        json_files = glob(os.path.join(episode_dir, '*_cleaned.json'))
        if not json_files:
            continue
        json_path = json_files[0]
        save_dir = os.path.join(save_root, episode_name)
        print(f'Processing {episode_dir}, saving camera episodes in {save_dir}')
        create_camera_episodes(episode_dir, json_path, save_dir, episode_name=episode_name)
    # Optionally save the instruction mapping
    instructions_save_path = os.path.join(save_root, 'all_folder_instructions.json')
    with open(instructions_save_path, 'w') as f:
        json.dump({episode_name: language_instructions for episode_name, language_instructions in LANGUAGE_INSTRUCTIONS.items()}, f, indent=2)
    print(f'Saved all folder instructions to {instructions_save_path}')

if __name__ == '__main__':
    main()

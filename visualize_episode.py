import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from PIL import Image

def visualize_episode(np_file):
    episode = np.load(np_file, allow_pickle=True)
    print(f"Loaded episode from {np_file} with {len(episode)} frames.")
    for i, frame in enumerate(episode):
        print(f"Frame {i}:")
        print(f"  State: {frame['state']}")
        print(f"  Action: {frame['action']}")
        print(f"  Language instruction: {frame['language_instruction']}")
        # Visualize RGB image and depth if available
        if frame['image'] is not None:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(frame['image'])
            plt.title(f'RGB Frame {i}')
            plt.axis('off')
            if frame['depth_image'] is not None:
                plt.subplot(1, 2, 2)
                plt.imshow(frame['depth_image'], cmap='gray')
                plt.title(f'Depth Frame {i}')
                plt.axis('off')
            plt.show()
        else:
            print("  No RGB image available for this frame.")
        # Only show first 5 frames for sanity
        if i >= 50:
            break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <episode_npy_file>")
        sys.exit(1)
    np_file = sys.argv[1]
    visualize_episode(np_file)

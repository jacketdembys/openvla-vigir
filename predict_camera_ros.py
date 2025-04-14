import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configuration
RESIZE_IMAGE = False
VISUALIZE_EACH_FRAME = False

if __name__ == '__main__':
    print("\n\n==> OpenVLA Camera Feed Prediction ...")

    # Find torch device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'
    print(f"==> Running on: {device_name}")

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        resume_download=None
    ).to(device)

    print("\n==> Using camera feed")
    cam = cv2.VideoCapture(0)  # its on /dev/video0

    if RESIZE_IMAGE:
        width = 224
        height = 224
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Image shape: [{w},{h}]")
    f = 0
    action_list = []

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Convert the OpenCV frame (BGR) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Optionally visualize the frame
        if VISUALIZE_EACH_FRAME:
            plt.imshow(image)
            plt.axis("off")
            plt.show()

        # Perform inference
        current_prompt = 'Pick up the water bottle.'
        prompt = f"In: What action should the robot take to {current_prompt}?\nOut:"
        print(f"Current prompt {f} for frame {f}: {current_prompt}")
        print(f"Prompt given to the robot: {prompt}")

        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        action[-1] = np.round(action[-1])

        np.set_printoptions(suppress=True)
        print("[" + " ".join(f"{x:.8f}" for x in action) + "]")
        print("\n")
        action_list.append(action)

        f += 1
        # Optional: break after a certain number of frames
        # if f == 200:
        #     np.save("grabbing_r10.npy", action_list)
        #     break

    cam.release()
    cv2.destroyAllWindows()

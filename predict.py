import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import wandb
import tensorflow_datasets as tfds
import tqdm





USE_CAMERA_FEED           = False           # Use feed from camera: True / Use example data: False
VISUALIZE_EXAMPLE_EPISODE = False           # From example data
USE_WAND = True                             # Use wandb
VISUALIZE_EACH_FRAME = False 
RESIZE_IMAGE = False 
DATASET = "bridge"   # jaco_play, droid           




if __name__ == '__main__':
    print("\n\n==> Test OpenVLA ...")


    if USE_WAND:
        wandb.init(
            entity="jacketdembys",
            project="vigir_vla",
            group="model_openvla", 
            job_type=DATASET + "_dataset",
            name="episode_1"
        )

        columns = ["Image", "Delta_X", "Delta_Y", "Delta_Z", "Delta_Ro", "Delta_Pi", "Delta_Yaw", "Gripper_Signal"]
        table = wandb.Table(columns=columns)

        #gcolumns = ["Image", "gDelta_X", "gDelta_Y", "gDelta_Z", "gDelta_Ro", "gDelta_Pi", "gDelta_Yaw", "gGripper_Signal"] 
        gtable = wandb.Table(columns=columns)


    # Find torch device 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    device_name = torch.cuda.get_device_name(device)
    print("==> Running on: {}".format(device_name))

    
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        resume_download=None
    ).to("cuda:0")
    

    if USE_CAMERA_FEED:
        print("\n==> Using camera feed")

        cam = cv2.VideoCapture(0) # its on /dev/video0

        if RESIZE_IMAGE:
            width = 224 #1280
            height = 224 #960
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Image shape: [{w},{h}]")
        f = 0
        action_list = []

        while True:
            
            ret, frame = cam.read()

            # Safety check to make sure we successfully read a frame
            if not ret:
                print("Failed to grab frame")
                break
            # Convert the OpenCV frame (BGR) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print(frame_rgb.shape)
            # print(frame)

            # Now convert the np array to PIL Image
            image = Image.fromarray(frame_rgb)
            
            if VISUALIZE_EXAMPLE_EPISODE:
                cv2.imshow('Cam feed', frame)


                if cv2.waitKey(1) == ord('q'):
                    break


            # Perform inference
            current_prompt = 'Pick up the water bottle.'
            prompt = f"In: What action should the robot take to {current_prompt}?\nOut:"
            print(f"Current prompt {f} for frame {f}: {current_prompt}")
            print(f"Prompt given to the robot: {prompt}")
        
            
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            action[-1] = np.round(action[-1])

            np.set_printoptions(suppress=True)
            #print(action)
            print("[" + " ".join(f"{x:.8f}" for x in action) + "]")
            print("\n")
            action_list.append(action)
            if f == 200:
               np.save("grabbing_r10.npy", action_list)
               break
            
            if USE_WAND:
                table.add_data(wandb.Image(image), *action)
                gtable.add_data(wandb.Image(image), *action)
            
            
            if VISUALIZE_EACH_FRAME:
                plt.imshow(image)
                plt.axis("off")  
                plt.show()

            f += 1
        


        if USE_WAND:    
            wandb.log({"Results Table": table})
            wandb.finish()




        cam.release()
        cv2.destroyAllWindows()



    else:

        if DATASET == "jaco_play":

            print("\n==> Using test data from CLVR Jaco Play Dataset")

            data_path = "/home/retina/dembysj/Dropbox/research/VLA/openvla/data/test_data_jaco_play.h5"
            data = {}
            with h5py.File(data_path) as F:
                for key in F.keys():
                    data[key] = np.array(F[key])

            print("Dataset shapes: ")
            for key in data:
                print(f"{key}: {data[key].shape}")

            
            if VISUALIZE_EXAMPLE_EPISODE:
                image_seq = None
                gt_ee_pose = None
                for i in range(data['front_cam_ob'].shape[0])[::5]:
                    if image_seq is None:
                        image_seq = data['front_cam_ob'][i]
                        #gt_ee_pose = data['ee_cartesian_pos_ob'][i]
                    else:
                        image_seq = np.concatenate((image_seq, data['front_cam_ob'][i]), axis=1)
                        #gt_ee_pose_sequence = np.concatenate((gt_ee_pose, data['ee_cartesian_pos_ob'][i]), axis=1)
                    if data['terminals'][i]: break

                
                
                
                print(f"\n\nCurrent prompt = {data['prompts'][0]}")
                print(f"Sequence length: {image_seq.shape}")
                print(f"Sequence length gt: {gt_ee_pose_sequence.shape}")

                plt.figure(figsize=(20, 30))
                plt.imshow(image_seq[..., ::-1])
                plt.show()

                #sys.exit()

                
            print(f"\n\nNumber of frames in the video: {data['front_cam_ob'].shape[0]}")
            #sys.exit()
            for i in range(data['front_cam_ob'].shape[0]): #[::50]:
                current_frame = data['front_cam_ob'][i,:,:,:]
                current_target_action = data['ee_cartesian_pos_ob'][i]
                current_target_action[-1] = np.round(current_target_action[-1])
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(current_frame)

                current_prompt = data['prompts'][i]
                prompt = f"In: What action should the robot take to {current_prompt}?\nOut:"
                print(f"Current prompt {i} for frame {i}: {current_prompt}")
                print(f"Prompt given to the robot: {prompt}")
            
                
                inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
                #action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                action = vla.predict_action(**inputs, unnorm_key="jaco_play", do_sample=False)
                action[-1] = np.round(action[-1])

                np.set_printoptions(suppress=True)
                #print(action)
                print("Prediction:\t[" + " ".join(f"{x:.8f}" for x in action) + "]")
                print("Target:\t\t[" + " ".join(f"{x:.8f}" for x in current_target_action) + "]")
                print("\n")
                
                #sys.exit()

                #print(action)
                #print(current_target_action)

                #sys.exit()
                if USE_WAND:
                    table.add_data(wandb.Image(image), *action)
                    gtable.add_data(wandb.Image(image), *current_target_action)
                    
                if VISUALIZE_EACH_FRAME:
                    plt.imshow(image)
                    plt.axis("off")  
                    plt.show()

            if USE_WAND:
                wandb.log({
                    "Results Table": table,
                    "GT Table": gtable
                            })
                #wandb.log()
                wandb.finish()



        elif DATASET == "droid":
            
            print("\n==> Using test data from local Jaco Play Dataset")

            """
            DATASET_NAMES = ['droid_100']
            DOWNLOAD_DIR = './data/1.0.0'

            print(f"Loading {len(DATASET_NAMES)} dataset from {DOWNLOAD_DIR}.")
            for dataset_name in tqdm.tqdm(DATASET_NAMES):
                data = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)

            print(data)
            """

            dataset_path = "data"
            dataset_name = "droid_100"
            ds = tfds.load(dataset_name, data_dir=dataset_path, split="train")
            images = []
            for episode in ds.shuffle(10, seed=0).take(1):
                for i, step in enumerate(episode["steps"]):

                    print(f"\nProcessing frame {i}")

                    # get image
                    image = step["observation"]["exterior_image_1_left"]
                    image = image.numpy()
                    image = Image.fromarray(image)
                    #print(image)

                    # get target pose
                    action_pose = step["observation"]["cartesian_position"]
                    action_pose = action_pose.numpy().tolist()
                    #print(action_pose)

                    # get target gripper
                    action_gripper = step["observation"]["gripper_position"]
                    action_gripper = action_gripper.numpy().tolist()
                    #print(action_gripper)

                    # build entire target pose action
                    current_target_action = action_pose + action_gripper
                    #print(action)

                    # get prompt
                    current_prompt = step["language_instruction"]
                    current_prompt = current_prompt.numpy().decode("utf-8")
                    #print(prompt)

                    prompt = f"In: What action should the robot take to {current_prompt}?\nOut:"
                    print(f"Current prompt {i} for frame {i}: {current_prompt}")
                    print(f"Prompt given to the robot: {prompt}")

                    # Process model
                    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
                    #action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                    action[-1] = np.round(action[-1])

                    np.set_printoptions(suppress=True)
                    #print(action)
                    print("Prediction:\t[" + " ".join(f"{x:.8f}" for x in action) + "]")
                    print("Target:\t\t[" + " ".join(f"{x:.8f}" for x in current_target_action) + "]")
                    #print("\n")
                    
                    #sys.exit()

                    #print(action)
                    #print(current_target_action)

                    #sys.exit()
                    
                    if USE_WAND:
                        table.add_data(wandb.Image(image), *action)
                        gtable.add_data(wandb.Image(image), *current_target_action)
                    
                    if VISUALIZE_EACH_FRAME:
                        plt.imshow(image)
                        plt.axis("off")  
                        plt.show()

            if USE_WAND:
                wandb.log({
                    "Predictions Table": table,
                    "Targets Table": gtable
                            })
                #wandb.log()
                wandb.finish()

            # Delete tensorflow variable and get out of the script
            del ds, image, action_pose, action_gripper, prompt

                    
            
       
        elif DATASET == "bridge":
            
            dataset_path = "data"
            dataset_name = "bridge_orig"
            ds = tfds.load(dataset_name, data_dir=dataset_path, split="train")
            images = []

            for episode in ds.shuffle(buffer_size=10).take(1):
                for i, step in enumerate(episode["steps"]):

                    print(f"\nProcessing frame {i}")

                    # get image
                    image = step["observation"]["image_0"]
                    image = image.numpy()
                    image = Image.fromarray(image)
                    #print(image)

                    # get entire target pose action
                    current_target_action = step["action"]
                    current_target_action = current_target_action.numpy().tolist()
                    #print(current_target_action)

                    # get prompt
                    current_prompt = step["language_instruction"]
                    current_prompt = current_prompt.numpy().decode("utf-8")                  

                    prompt = f"In: What action should the robot take to {current_prompt}?\nOut:"
                    print(f"Current prompt {i} for frame {i}: {current_prompt}")
                    print(f"Prompt given to the robot: {prompt}")
                    

                    # Process model
                    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
                    #action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
                    action[-1] = np.round(action[-1])

                    np.set_printoptions(suppress=True)
                    #print(action)
                    print("Prediction:\t[" + " ".join(f"{x:.8f}" for x in action) + "]")
                    print("Target:\t\t[" + " ".join(f"{x:.8f}" for x in current_target_action) + "]")
                    #print("\n")
                    
                    #sys.exit()

                    #print(action)
                    #print(current_target_action)

                    #sys.exit()
                    
                    if USE_WAND:
                        table.add_data(wandb.Image(image), *action)
                        gtable.add_data(wandb.Image(image), *current_target_action)
                    
                    if VISUALIZE_EACH_FRAME:
                        plt.imshow(image)
                        plt.axis("off")  
                        plt.show()

            if USE_WAND:
                wandb.log({
                    "Predictions Table": table,
                    "Targets Table": gtable
                            })
                #wandb.log()
                wandb.finish()

            # Delete tensorflow variable and get out of the script
            del ds, image, current_target_action, current_prompt


        #sys.exit()



    
    




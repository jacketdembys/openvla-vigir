import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
# ROS 2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import time

# Configuration
RESIZE_IMAGE = False
VISUALIZE_EACH_FRAME = False
PUBLISH_RATE_HZ = 0.2  # Set desired publish rate in Hz (actions per second)
OPENVLA_MODEL_PATH = "/root/huggingface_models/openvla-7b"
UNNORM_KEY = "bridge_orig"

class OpenVLAImagePredictor(Node):
    def __init__(self):
        super().__init__('openvla_image_predictor')
        self.bridge = CvBridge()
        self.processor = AutoProcessor.from_pretrained(OPENVLA_MODEL_PATH, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            OPENVLA_MODEL_PATH,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            resume_download=None
        ).to(self.get_device())
        self.f = 0
        self.action_list = []
        self.publisher_ = self.create_publisher(Float32MultiArray, 'predicted_actions', 10)
        self.publish_rate = PUBLISH_RATE_HZ
        self.last_pub_time = time.time()
        self.subscription = self.create_subscription(
            RosImage,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to /camera/camera/color/image_raw')
        self.get_logger().info('Publishing predicted actions to /predicted_actions')

    def get_device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def image_callback(self, msg):
        now = time.time()
        # Rate limiting
        if now - self.last_pub_time < 1.0 / self.publish_rate:
            return
        self.last_pub_time = now

        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Optionally resize
        if RESIZE_IMAGE:
            cv_image = cv2.resize(cv_image, (224, 224))
        # Convert to RGB and PIL Image
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Optionally visualize
        if VISUALIZE_EACH_FRAME:
            plt.imshow(image)
            plt.axis("off")
            plt.show()

        # Perform inference
        current_prompt = 'Pick up the yellow object.'
        prompt = f"In: What action should the robot take to {current_prompt}?\nOut:"
        self.get_logger().info(f"Current prompt {self.f} for frame {self.f}: {current_prompt}")
        self.get_logger().info(f"Prompt given to the robot: {prompt}")
        device = self.get_device()
        inputs = self.processor(prompt, image).to(device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
        action[-1] = np.round(action[-1])
        np.set_printoptions(suppress=True)
        print("[" + " ".join(f"{x:.8f}" for x in action) + "]\n")
        self.action_list.append(action)
        self.f += 1

        # Publish actions to topic
        action_msg = Float32MultiArray()
        action_msg.data = action.astype(np.float32).tolist()
        self.publisher_.publish(action_msg)


def main(args=None):
    rclpy.init(args=args)
    predictor = OpenVLAImagePredictor()
    print("\n\n==> OpenVLA Camera Feed Prediction (ROS 2)...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'
    print(f"==> Running on: {device_name}")
    try:
        rclpy.spin(predictor)
    except KeyboardInterrupt:
        print('Shutting down predictor node.')
    predictor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

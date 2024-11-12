"""
    Author: Jason Hughes
    Date: November 2024

    About: A ROS 2 node for publishing depth images from a 
    a single image input, using Apple's ml-depth-pro

"""
import cv2
import rclpy
import depth_pro

import numpy as np

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from depth_pro_ros.intrinsics import CameraIntrinsics
from depth_pro.depth_pro import DepthProConfig

class DepthProInference(Node):

    def __init__(self) -> None:
        super().__init__('depth_pro_inference')

        self.declare_parameter("intrinsics_path", "./config/blackfly.yaml")
        self.declare_parameter("model_path", "./models/depth_pro.pt")

        intrinsics_path = self.get_parameter("intrinsics_path").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        
        self.timer_ = self.create_timer(1, self.infernceCallback)

        self.img_sub_ = self.create_subscription(Image, '/image', self.imageCallback, 1)
        self.cimg_sub_ =self.create_subscription(CompressedImage, '/image/compressed', self.compressedImageCallback, 1)

        config = DepthProConfig(patch_encoder_preset="dinov2l16_384",
                                image_encoder_preset="dinov2l16_384",
                                checkpoint_uri=model_path,
                                decoder_features=256,
                                use_fov_head=True,
                                fov_encoder_preset="dinov2l16_384")

        self.model_, self.transform_ = depth_pro.create_model_and_transforms(config = config)

        self.model_.eval()

        self.bridge_ = CvBridge()

        self.intrinsics_ = CameraIntrinsics(intrinsics_path)

        self.img_ = None

        self.get_logger().info("Depth Pro Initialized")

    def infernceCallback(self) -> None:
        if self.img_ is not None:
            img = self.transform_(self.img)
            pred = self.model_.infer(img, self.intrinsics_.focal_length.x)

            depth = pred["depth"]

            # publish it as depth image?? and point cloud

    def imageCallback(self, msg : Image) -> None:
        try:
            self.img_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return 

    def compressedImageCallback(self, msg : CompressedImage) -> None:
        try:
            self.img_ = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed image: {e}")
            return 

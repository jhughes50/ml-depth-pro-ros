"""
    Author: Jason Hughes
    Date: November 2024

    About: A ROS 2 node for publishing depth images from a 
    a single image input, using Apple's ml-depth-pro

"""
import cv2
import rclpy
import depth_pro
import torch
import struct

import numpy as np

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from depth_pro_ros.intrinsics import CameraIntrinsics
from depth_pro.depth_pro import DepthProConfig
#from depth_image_proc.depth_conversions import convert_depth_image_to_pointcloud2

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1  # Only keep the latest message
)

class DepthProInference(Node):

    def __init__(self) -> None:
        super().__init__('depth_pro_inference')

        self.declare_parameter("intrinsics_path", "./config/blackfly.yaml")
        self.declare_parameter("model_path", "./models/depth_pro.pt")

        intrinsics_path = self.get_parameter("intrinsics_path").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        
        self.sub_group_ = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.timer_group_ = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

        self.timer_ = self.create_timer(1, self.infernceCallback, callback_group=self.timer_group_)

        self.img_sub_ = self.create_subscription(Image, '/image', self.imageCallback, 1, callback_group=self.sub_group_)
        self.cimg_sub_ =self.create_subscription(CompressedImage, '/image/compressed', self.compressedImageCallback, qos_profile, callback_group=self.sub_group_)

        self.pc_pub_ = self.create_publisher(PointCloud2, '/cloud', 1)

        config = DepthProConfig(patch_encoder_preset="dinov2l16_384",
                                image_encoder_preset="dinov2l16_384",
                                checkpoint_uri=model_path,
                                decoder_features=256,
                                use_fov_head=True,
                                fov_encoder_preset="dinov2l16_384")

        self.model_, self.transform_ = depth_pro.create_model_and_transforms(config = config)

        self.model_.eval()
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_.to(self.device_)
        self.get_logger().info("Using device %s" %self.device_)

        self.bridge_ = CvBridge()

        self.intrinsics_ = CameraIntrinsics(intrinsics_path)

        self.img_ = None
        self.count_ = 0
        self.get_logger().info("Depth Pro Initialized")

    def infernceCallback(self) -> None:
        if self.img_ is not None:
            self.get_logger().info("starting inference")
            img = self.transform_(self.img_)
            #print("focal length: ", self.intrinsics_.focal_length.x)
            pred = self.model_.infer(img.to(self.device_), self.intrinsics_.focal_length.x_tensor.to(self.device_))

            depth = pred["depth"]
            self.get_logger().info("finished inference")
            # publish it as depth image?? and point cloud
            #self.get_logger().info("shape: %s" %depth.shape)
            #print(depth.shape)
            msg = self.imageToPointCloud(depth.cpu().numpy())
            self.pc_pub_.publish(msg)

            
    def imageToPointCloud(self, depth : np.ndarray) -> PointCloud2:
        height, width = depth.shape

        x = np.tile(np.arange(width), (height, 1))
        y = np.tile(np.arange(height), (width, 1)).T
        
        # Calculate 3D coordinates
        z = depth
        x = (x - self.intrinsics_.focal_length.cx) * z / self.intrinsics_.focal_length.x
        y = (y - self.intrinsics_.focal_length.cy) * z / self.intrinsics_.focal_length.y
        
        # Stack coordinates and reshape
        xyz = np.stack((x, y, z), axis=-1)
        points = xyz.reshape(-1, 3)
        #points = cv2.resize(points, (1024, 768), interpolation = cv2.INTER_LINEAR)
        # Remove invalid points (where depth is 0 or NaN)
        mask = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
        points = points[mask]


        msg = PointCloud2()
        msg.header.frame_id = "map"
        msg.height = 1 
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.point_step = 12  # 4 bytes * 3 fields (x, y, z)
        msg.row_step = msg.point_step * msg.width

        buffer = points.astype(np.float32).tobytes()

        expected_size = msg.width * msg.point_step
        actual_size = len(buffer)
        self.get_logger().info("expected size %i vs actual size %i" %(expected_size, actual_size))
        #for p in points:
        #    buffer.extend(struct.pack('fff', p[0], p[1], p[2]))

        msg.data = buffer

        return msg

    def imageCallback(self, msg : Image) -> None:
        try:
            self.img_ = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return 

    def compressedImageCallback(self, msg : CompressedImage) -> None:
        self.get_logger().info("image recieved: %i" %self.count_ )
        self.count_ += 1
        try:
            self.img_ = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed image: {e}")
            return 

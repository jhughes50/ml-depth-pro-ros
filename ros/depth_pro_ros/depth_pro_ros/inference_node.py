"""
    Author: Jason Hughes
    Date: November 2024

    About: start up strict for depth pro ros inference 

"""
import rclpy
from depth_pro_ros.inference import DepthProInference

def main(args=None) -> None:
    rclpy.init(args=args)

    dpr = DepthProInference()

    rclpy.spin(dpr)

    dpr.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

"""
    Author: Jason Hughes
    Date: November 2024

    About: start up strict for depth pro ros inference 

"""
import rclpy
import threading

from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from depth_pro_ros.inference import DepthProInference

def main(args=None) -> None:
    rclpy.init(args=args)

    node = DepthProInference()
    executor = MultiThreadedExecutor()

    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    try:
        executor_thread.start()
    finally:
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

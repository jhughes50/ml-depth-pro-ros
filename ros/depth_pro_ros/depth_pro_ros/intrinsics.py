"""
    Author: Jason Hughes
    Date: November 2024

    About: object to handle instrinsics from kalibr
"""

import yaml

class FocalLength:

    def __init__(self, x : float, y : float, cx : float = 0.0, cy : float = 0.0) -> None:
        self.fx_ = x
        self.fy_ = y

        self.cx_ = cx
        self.cy_ = cy

    @property
    def x(self) -> float:
        return self.fx_

    @property
    def y(self) -> float:
        return self.fy_

    @property
    def cx(self) -> float:
        return self.cx_

    @property
    def cy(self) -> float:
        return self.cy_

class Resolution:

    def __init__(self, x : int, y : int) -> None:
        self.x_ = x
        self.y_ = y

    @property
    def x(self) -> int:
        return self.x_
    
    @property
    def width(self) -> int:
        return self.x_

    @property
    def y(self) -> int:
        return self.y_

    @property
    def height(self) -> int:
        return self.y_


class CameraIntrinsics:

    def __init__(self, path : str) -> None:

        with open(path, "r") as file:
            self.data_ = yaml.safe_load(file)
        
        self.data_ = self.data_["cam0"]

        self.focal_length = FocalLength(self.data_["intrinsics"][0], self.data_["intrinsics"][1], self.data_["intrinsics"][2], self.data_["intrinsics"][3])

        self.model = self.data_["distortion_model"]

        self.resolution = Resolution(self.data_["resolution"][0], self.data_["resolution"][1])

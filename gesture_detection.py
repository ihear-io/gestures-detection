import math
from typing import *
import os
import cv2
import argparse
import numpy as np
from utils import import_open_pose
import env_vars

_op = import_open_pose()
_op_wrapper = _op.WrapperPython()

params = {
    "model_folder": env_vars.MODEL_LOC,
    "model_pose": "COCO",
    "number_people_max": 1,
    "net_resolution": "-1x64"
}


def _distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def _hands_rectangles(images: List, err_thresh: float, debug=False) -> List[List[_op.Rectangle]]:
    _op_wrapper.configure(params)
    _op_wrapper.start()

    if debug:
        cv2.namedWindow("body pose", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions

    hands = []
    for img in images:
        datum = _op.Datum()
        datum.cvInputData = img
        _op_wrapper.emplaceAndPop([datum])

        x_left, y_left, _ = datum.poseKeypoints[0][7]  # left wrist
        x_right, y_right, _ = datum.poseKeypoints[0][4]  # right wrist

        x_elbow_l, y_elbow_l, score_l = datum.poseKeypoints[0][6]
        x_elbow_r, y_elbow_r, score_r = datum.poseKeypoints[0][3]

        len_hand_left = _distance(x_left, y_left, x_elbow_l, y_elbow_l)
        len_hand_right = _distance(x_right, y_right, x_elbow_r, y_elbow_r)

        x_left_shifted = max(0, x_left - len_hand_left)
        y_left_shifted = max(0, y_left - len_hand_left)
        x_right_shifted = max(0, x_right - len_hand_right)
        y_right_shifted = max(0, y_right - len_hand_right)

        rect_len_left = 2 * max(x_left - x_left_shifted, y_left - y_left_shifted) \
            if score_l > err_thresh else 0
        rect_len_right = 2 * max(x_right - x_right_shifted, y_right - y_right_shifted) \
            if score_r > err_thresh else 0

        if debug:
            print((rect_len_right, rect_len_left))
            cv2.imshow("body pose", datum.cvOutputData)
            cv2.waitKey(0)

        hand_rectangles = [
            [
                _op.Rectangle(x_left_shifted, y_left_shifted, rect_len_left, rect_len_left),  # left hand
                _op.Rectangle(x_right_shifted, y_right_shifted, rect_len_right, rect_len_right),  # right hand
            ]
        ]
        hands.append(hand_rectangles)
    cv2.destroyAllWindows()
    return hands


def hand_keypoints(images: List, debug=False, err_thresh: float = 0.1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Uses body pose estimation to estimate lwrist & rwrist positions,
    from which we use `openpose` to obtain hand keypoints vectors.

    :returns a pair of (left_hand_keypoints: ndarray, right_hand_keypoints: ndarray).

    :param images: a list of images to process.
    :param debug: whether or not to display results via `opencv` methods.
    :param err_thresh: indicates how much score is considered false positive.
    """

    hands = _hands_rectangles(images, err_thresh, debug)
    hand_params = {
        "model_folder": env_vars.MODEL_LOC,
        "model_pose": "COCO",
        "number_people_max": 1,
        "hand": True,
        "hand_detector": 2,
        "body": 0,
    }
    _op_wrapper.configure(hand_params)
    _op_wrapper.start()
    keypoints: List[Tuple[np.ndarray, np.ndarray]] = []

    if debug:
        cv2.namedWindow("hand key points", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions

    for i, img in enumerate(images):
        datum = _op.Datum()
        datum.cvInputData = img
        datum.handRectangles = hands[i]
        _op_wrapper.emplaceAndPop([datum])

        ls = datum.handKeypoints[0][0]  # left
        rs = datum.handKeypoints[1][0]  # right

        # sanitization
        l_avg = sum(map(lambda y: y[2], ls)) / 21.0
        r_avg = sum(map(lambda y: y[2], rs)) / 21.0
        if l_avg < err_thresh:
            ls *= (0.0, 0.0, 1.0)
        if r_avg < err_thresh:
            rs *= (0.0, 0.0, 1.0)

        if debug:
            print("left avg: ", l_avg)
            print("right avg: ", r_avg)
            print("Left hand keypoints:\n", ls)
            print("Right hand keypoints:\n", rs)
            cv2.imshow("hand key points", datum.cvOutputData)
            cv2.waitKey(0)

        keypoints.append((ls, rs))

    cv2.destroyAllWindows()
    return keypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    hand_keypoints([cv2.imread(args[0].image_path)], True)

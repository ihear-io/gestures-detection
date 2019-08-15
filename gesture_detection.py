from typing import *
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


def _hands_rectangles(images: List, debug=False) -> List[List[_op.Rectangle]]:
    _op_wrapper.configure(params)
    _op_wrapper.start()
    hands = []
    for img in images:
        datum = _op.Datum()
        datum.cvInputData = img
        _op_wrapper.emplaceAndPop([datum])

        height = np.size(img, 0)
        width = np.size(img, 1)
        axis: int = width > height  # axis of the longest dimension.
        margin = np.size(img, axis) * 0.1
        length = np.size(img, axis) * 0.6

        if debug:
            cv2.imshow("body pose", datum.cvOutputData)

        x_left, y_left, _ = datum.poseKeypoints[0][7]  # left wrist
        x_right, y_right, _ = datum.poseKeypoints[0][4]  # right wrist

        hand_rectangles = [
            [
                _op.Rectangle(x_left - margin, y_left - margin, length, length),  # left hand
                _op.Rectangle(x_right - margin, y_right - margin, length, length),  # right hand
            ]
        ]
        hands.append(hand_rectangles)

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

    hands = _hands_rectangles(images, debug)
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
    for i, img in enumerate(images):
        datum = _op.Datum()
        datum.cvInputData = img
        datum.handRectangles = hands[i]
        _op_wrapper.emplaceAndPop([datum])

        ls = datum.handKeypoints[0][0]  # left
        rs = datum.handKeypoints[1][0]  # right

        l_avg = sum(map(lambda y: y[2], ls)) / 21.0
        print("left avg: ", l_avg)
        r_avg = sum(map(lambda y: y[2], rs)) / 21.0
        print("right avg: ", r_avg)

        # sanitization
        if l_avg < err_thresh:
            ls *= (0.0, 0.0, 1.0)
        if r_avg < err_thresh:
            rs *= (0.0, 0.0, 1.0)
        keypoints.extend((ls, rs))

        if debug:
            print("Left hand keypoints:\n", ls)
            print("Right hand keypoints:\n", rs)
            cv2.imshow("hand key points", datum.cvOutputData)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return keypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    hand_keypoints([cv2.imread(args[0].image_path)], True)

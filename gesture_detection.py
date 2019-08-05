import sys
import cv2
import argparse
import numpy as np
from utils import import_open_pose
import env_vars

op = import_open_pose()

params = {"model_folder": env_vars.MODEL_LOC, "model_pose": "COCO", "number_people_max": 1, "net_resolution": "-1x64"}


def hand_keypoints(img, debug=False, err_thresh=0.1, margin=250.0, l=450.0):
    """
    Uses body pose estimation to estimate lwrist & rwrist positions,
    from which we use `openpose` to obtain hand keypoints vectors. 
    \n
    Returns a pair of (left_hand_keypoints: ndarray, right_hand_keypoints: ndarray).\n
    \n
    params:\n
    `img`: the image.\n
    `debug`: whther or not to display results via `opencv` methods.\n
    `margin`: margin of each hand rectangle.\n
    `l`: squared rectangle length.\n
    """
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    datum = op.Datum()
    datum.cvInputData = img
    op_wrapper.emplaceAndPop([datum])

    if debug:
        cv2.imshow("body pose", datum.cvOutputData)

    x_left, y_left, _ = datum.poseKeypoints[0][7]  # left wrist
    x_right, y_right, _ = datum.poseKeypoints[0][4]  # righr wrist

    hand_rectangles = [
        [
            op.Rectangle(x_left - margin, y_left - margin, l, l),  # left hand
            op.Rectangle(x_right - margin, y_right - margin, l, l),  # right hand
        ]
    ]

    hand_params = {
        "model_folder": env_vars.MODEL_LOC,
        "model_pose": "COCO",
        "number_people_max": 1,
        "hand": True,
        "hand_detector": 2,
        "body": 0,
    }
    op_wrapper.configure(hand_params)
    op_wrapper.start()
    datum = op.Datum()
    datum.cvInputData = img
    datum.handRectangles = hand_rectangles
    op_wrapper.emplaceAndPop([datum])

    ls = datum.handKeypoints[0][0]  # left
    rs = datum.handKeypoints[1][0]  # right
    print(type(ls))
    l_avg = sum(map(lambda y: y[2], ls)) / 21.0
    print("left avg: ", l_avg)
    r_avg = sum(map(lambda y: y[2], rs)) / 21.0
    print("right avg: ", r_avg)

    # sanitization
    if l_avg < err_thresh:
        ls = np.array(list(map(lambda ns: np.array([0.0, 0.0, ns[2]]), ls)))
    if r_avg < err_thresh:
        rs = np.array(list(map(lambda ns: np.array([0.0, 0.0, ns[2]]), rs)))

    print("Left hand keypoints:\n", ls)
    print("Right hand keypoints:\n", rs)

    if debug:
        cv2.imshow("hand key points", datum.cvOutputData)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return ls, rs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    hand_keypoints(cv2.imread(args[0].image_path), True)

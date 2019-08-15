import os

OPEN_POSE_LOC_UBUNTU = "/usr/local/python"
OPEN_POSE_LOC_WIN = "INSERT OPEN POSE PATH HERE"  # TODO

OPEN_POSE_LOC = OPEN_POSE_LOC_WIN if os.name == 'nt' else OPEN_POSE_LOC_UBUNTU

MODEL_LOC = "/home/mhashim6/dev/openpose/models/"

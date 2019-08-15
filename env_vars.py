import os

OPEN_POSE_LOC_UBUNTU = "/usr/local/python"
OPEN_POSE_LOC_WIN = "INSERT OPEN POSE PATH HERE"  # TODO

if os.name == 'nt':  # windows
    OPEN_POSE_LOC = OPEN_POSE_LOC_WIN
else:
    OPEN_POSE_LOC = OPEN_POSE_LOC_UBUNTU

MODEL_LOC = "/home/mhashim6/dev/openpose/models/"

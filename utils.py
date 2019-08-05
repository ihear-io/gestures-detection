from sys import platform, path
from env_vars import OPEN_POSE_LOC


def import_open_pose():
    """
    Assuming openpose path is already set in [env_vars.py].
    """
    path.insert(1, OPEN_POSE_LOC)
    try:
        # Windows Import
        if platform == "win32":
            import pyopenpose
        else:  # Ubuntu
            from openpose import pyopenpose

        return pyopenpose
    except ImportError as e:
        print(
            "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python "
            "script in the right folder? "
        )
        raise e

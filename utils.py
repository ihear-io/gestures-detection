import os
from sys import platform


def import_open_pose():
    """
    Assuming openpose is already in your PATH/PYTHONPATH, or local env.
    """

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

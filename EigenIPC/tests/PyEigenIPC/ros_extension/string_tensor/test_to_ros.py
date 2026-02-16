from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_to_ros_srvr import main


if __name__ == "__main__":
    main()

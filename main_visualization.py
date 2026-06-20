"""Compatibility wrapper.

Allows running `python main_visualization.py` as an alias for
`python main_visualize.py`.
"""

import os
import runpy


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    runpy.run_path(os.path.join(here, "main_visualize.py"), run_name="__main__")

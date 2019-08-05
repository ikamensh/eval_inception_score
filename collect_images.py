import os

import imageio
import numpy as np


def gather_np(folder):
    assert os.path.isdir(folder)
    result = []

    for file in os.listdir(folder):
        try:
            img = imageio.imread(os.path.join(folder, file))
            result.append(np.array(img))
        except Exception as e:
            print(f"Failed to read {file} as an image:", e)

    return result





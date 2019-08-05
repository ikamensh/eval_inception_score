import os
import time

from inc_score import inception_score
from collect_images import gather_np

from typing import Callable

def plot_inception(generated_dir: str, step_from_filename: Callable[[str], int]):
    scan_dir = os.path.join(generated_dir, "output")
    from ilya_ezplot import Metric, ez_plot
    m = Metric('step', 'inception_score')
    try:
        for file in os.listdir(scan_dir):
            current = os.path.join(scan_dir, file)
            if os.path.isdir(current):
                t = time.time()
                step = step_from_filename(file)
                images = gather_np(current)
                inc_score = inception_score(images[:400])
                m.add_many(step, inc_score)
                print(step, inc_score, f'evaluated in {time.time() - t:.1f}s')
    finally:
        ez_plot(m, os.path.join(generated_dir, 'plots'))






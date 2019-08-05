import os
from plot_dir import plot_inception

if __name__ == '__main__':
    step_from_name = lambda file: int(file.replace('step_', ''))

    dir_mid = os.path.join('E:', "generated_mid_models")
    plot_inception(dir_mid, step_from_name)

    dir_big = os.path.join('E:', "generated_big_models")
    plot_inception(dir_big, step_from_name)
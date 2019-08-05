import  os

import numpy as np
import imageio

from collect_images import gather_np

def test_collects(tmpdir):
    img = np.ones([100,100,3])
    imageio.imsave(os.path.join(tmpdir, f"{1}.jpg"), img)
    imageio.imsave(os.path.join(tmpdir, f"{2}.jpg"), img)

    imgs = gather_np(tmpdir)
    assert len(imgs) == 2



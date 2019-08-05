import numpy as np

from inc_score import inception_score


def test_diversity_counts():
    random = [np.random.randint(0, 255, [299, 299, 3], dtype='uint8') for _ in range(2)]

    fake_image_black = np.zeros([100, 100, 3]) + 11
    fake_image_white = 200 * np.ones_like(fake_image_black)

    copies = [fake_image_black] * 2
    different = [fake_image_black, fake_image_white]

    ic_random = inception_score(random)
    ic_black = inception_score(copies)
    ic_mix = inception_score(different)

    print(ic_random, ic_black, ic_mix)

    assert ic_mix > ic_random
    assert ic_random > ic_black






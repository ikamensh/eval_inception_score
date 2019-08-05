import os.path
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import math
import sys

print(tf.__version__)

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
_softmax = None

from typing import List


def inception_score(images: List[np.ndarray]):
    """
    Compute inception score.
    :param images: list of images. Each of elements should be a
    numpy array with values ranging from 0 to 255.
    :return: List of scores - given many images, they are split into batches and score is reported per batch.
    """
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray), type(images[0])
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    print(f'\ncomputing inception score on {len(images)} images of shape {images[0].shape}')
    inps = []
    splits = max(1, int(len(images) ** (1/2) // 50))
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    batch_size = 1
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(batch_size)))
        for i in range(n_batches):
            inp = inps[(i * batch_size):min((i + 1) * batch_size, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(_softmax, {'ExpandDims:0': inp})
            # print(pred.shape)
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))) # shape [BATCH, 1008]
            kl = np.mean(np.sum(kl, 1)) # shape -> [BATCH] -> []
            scores.append(np.exp(kl))
        return scores


def _init_inception():
    global _softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        _softmax = tf.nn.softmax(logits)


if _softmax is None:
    _init_inception()


if __name__ == "__main__":

    random = [np.random.randint(0, 255, [299,299,3], dtype='uint8') for _ in range(200)]
    print(inception_score(random))

    fake_image_black = np.zeros([100, 100, 3]) + 11
    fake_image_white = 200 * np.ones_like(fake_image_black)

    copies = [fake_image_black] * 200
    different = [fake_image_black, fake_image_white] * 100

    print(inception_score(copies))
    print(inception_score(different))

    # fake_image_black = np.zeros([32, 32, 3])
    # fake_image_white = 200 * np.ones_like(fake_image_black)
    #
    # copies = [fake_image_black] * 200
    # different = [fake_image_black, fake_image_white] * 10
    #
    # print(inception_score(copies))
    # print(inception_score(different))





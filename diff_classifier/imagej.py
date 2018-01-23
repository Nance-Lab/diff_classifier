import pandas as pd
import numpy as np
import skimage.io as sio


def partition_im(tiffname, irows=4, icols=4, ires=512):
    """
    partition_im(tiffname, irows=int, icols=int, ires=int)

    Partitions a 2048x2044 image into irows x icols images of size ires x ires and saved them.

    Parameters
    ----------
    tiffname : string
        Location of input image to be partitioned.
    irows : int
        Number of rows of size ires pixels to be partitioned from source image.
    icols : int
        Number of columns of size ires pixels to be partitioned from source image.
    ires : int
        Output images are of size ires x ires pixels.

    Examples
    ----------
    >>> partition_im('your/sample/image.tif', irows=8, icols=8, ires=256)

    """
    test = sio.imread(tiffname)
    test2 = np.zeros((test.shape[0], 2048, 2048), dtype=test.dtype)
    test2[:, 0:2044, :] = test

    new_image = np.zeros((test.shape[0], ires, ires), dtype=test.dtype)

    for row in range(irows):
        for col in range(icols):
            new_image = test2[:, row*ires:(row+1)*ires, col*ires:(col+1)*ires]
            sio.imsave(tiffname.split('.tif')[0] + '_%s_%s.tif' % (row, col), new_image)

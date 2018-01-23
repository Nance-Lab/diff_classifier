import numpy as np
import os.path as op
import skimage.io as sio
import subprocess
import tempfile
import diff_classifier as dc


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
            sio.imsave(tiffname.split('.tif')[0] + '_%s_%s.tif' % (row, col),
                       new_image)


def track(target, out_file, template=None, fiji_bin=None):
    """

    target : str
        Full path to a tif file to do tracking on.
        Can also be a URL (e.g., 'http://fiji.sc/samples/FakeTracks.tif')
    out_file : str
        Full path to a csv file to store the results.
    template : str, optional
        The full path of a template for tracking. Defaults to use
        `data/trackmate_template.py` stored in the diff_classifier source-code.
    """
    if template is None:
        template = op.join(op.split(dc.__file__)[0],
                           'data',
                           'trackmate_template.py')

    if fiji_bin is None:
        fiji_bin = op.join(op.expanduser('~'), 'Fiji.app/ImageJ-linux64')

    script = ''.join(open(template).readlines())
    tf = tempfile.NamedTemporaryFile(suffix=".py")
    fid = open(tf.name, 'w')
    fid.write(script.format(target_file=target))
    fid.close()
    cmd = "%s --ij2 --headless --run %s" % (fiji_bin, tf.name)
    print(cmd)
    sp = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    fid = open(out_file, 'w')
    fid.write(sp.stdout.decode())
    fid.close()

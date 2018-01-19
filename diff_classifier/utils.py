import pandas as pd
import numpy as np
import skimage.io as sio


def csv_to_pd(csvfname):
    csvfile = open(csvfname)

    line = 'test'
    counter = 0
    while line != 'Data starts here.\n':
        line = csvfile.readline()
        counter = counter + 1

    data = pd.read_csv(csvfname, skiprows=counter)
    data.sort_values(['Track_ID', 'Frame'], ascending=[1, 1])
    
    return data

def partition_im(tiffname, irows=4, icols=4, ires=512):
    test = sio.imread(tiffname)
    test2 = np.zeros((test.shape[0], 2048, 2048), dtype=test.dtype)
    test2[:, 0:2044, :] = test

    new_image = np.zeros((test.shape[0], ires, ires), dtype=test.dtype)

    for row in range(irows):
        for col in range(icols):
            new_image = test2[:, row*ires:(row+1)*ires, col*ires:(col+1)*ires]
            sio.imsave(tiffname.split('.tif')[0] + '_%s_%s.tif'%(row, col), new_image)
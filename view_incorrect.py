#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from astropy.io import fits

def convert(x):
    if x == 0:
        return "single"
    elif x == 1:
        return "multi"
    elif x == 2:
        return "deblend"
    else:
        return "error"

def main():
    exp_dir = "example_test"              # CHANGE THIS
    dataset_dir = '../../dataset'    # AND THIS
    filename = os.path.join(exp_dir, 'total_losses_pickle.pkl')
    with open(filename, 'rb') as pfile:
        model_out = pickle.load(pfile)
        for fits_file, true, guess in zip(model_out['val_files'], model_out['val_labels'], model_out['val_pred_labels']):
            if true != guess:
                endname = os.path.basename(fits_file)
                image_file = os.path.join(dataset_dir, endname)
                data = fits.getdata(image_file)
                plt.imshow(data)
                plt.title("Guess: {}, Truth: {}".format(convert(guess), convert(true)))
                plt.show()

if __name__ == "__main__":
    main()
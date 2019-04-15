import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import random



def main():
    # show_images()
    plot_results()


def plot_results():
    train_acc = []
    train_loss = []
    valid_acc = []
    valid_loss = []
    skip = True
    for line in open('../../mlpractical/exp_1/result_outputs/summary.csv', 'r').readlines():
        if skip:
            skip = False
            continue
        train_acc.append(float(line.split(',')[0]))
        train_loss.append(float(line.split(',')[1]))
        valid_acc.append(float(line.split(',')[2]))
        valid_loss.append(float(line.split(',')[3]))

    plt.plot(train_acc, label='Training')
    plt.plot(valid_acc, label='Validation')
    plt.title("Baseline network performance")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc=4)
    plt.show()
    # plt.savefig('accuracy.png')
    return

def show_images():
    counter = 0
    for file in os.listdir("../dataset"):
                 # "P200+55_ILTJ132119.65+563911.0_single.fits"
        if "deblend" in file:
            counter += 1
            image_file = "../dataset/" + file
            data = fits.getdata(image_file)
            print(data.shape)
            print(np.max(data), np.min(data))
            # exit()
            # plt.imshow(data)
            # plt.colorbar()
            plt.hist(data.flatten(), 255)
            plt.show()
            if counter > 5:
                return
    # random.seed(0xABCDEF)
    # data = os.listdir("../dataset")
    # random.shuffle(data)

if __name__ == "__main__":
    main()
    2
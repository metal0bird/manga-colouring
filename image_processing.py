import cv2
import os
from glob import glob
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import shutil
import scipy.signal as scig
from scipy import fftpack

# Helper Functions
# Creates a grid of subplots with specified rows and columns within a figure
def addAxis(thisfig, n1, n2):
    axlist = []
    for i in range(n1 * n2):
        axlist.append(thisfig.add_subplot(n1, n2, i + 1))
    return np.array(axlist)

# Removes ticks and labels from a list of axes for cleaner visualization
def groupFormat(axisList):
    # remove ticks and labels on all axis
    for ax in axisList:
        ax.set_xticks([])
        ax.set_yticks([])

# Deletes and recreates a folder to ensure a clean working directory
def delFold_RemakeFold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return

# Main Utility Functions
# Loads an image from a path, resizes it to 256x256, and returns it as an RGB NumPy array
def read_and_reshape(imgpath):
    # This function will load in a single image given the image folder path
    # and will then load, resize to 256x256
    # return the loaded, colored image as np array

    readImage = cv2.imread(imgpath)
    readImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
    readImage = np.array(cv2.resize(readImage, (256, 256)))

    return readImage

# Converts a colored image to a sketch using adaptive thresholding
def cvSketchify(image_np, blockSize=7):
    # This code takes in a single colored image and first converts
    # it to grayscale. After, it then uses CV edge detection to
    # convert to sketches
    # Play around with the sketchify settings until it looks right
    edgetype = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    #edgetype = cv2.ADAPTIVE_THRESH_MEAN_C
    grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    sketchified = np.array(
        cv2.adaptiveThreshold(grayscale,
                              255,
                              edgetype,
                              cv2.THRESH_BINARY,
                              blockSize=blockSize,
                              C=2)) / 255.0

    return sketchified

# Calculates a complexity metric for a black-and-white image, representing the percentage of black pixels
def complexityMetric(bwImage):
    return (256 * 256 - np.sum(np.sum(bwImage))) / (256 * 256) * 100

# Generates color cues from a colored image using convolution with a kernel
def genColorCues(cimg):
    out = scig.fftconvolve(cimg, kernel3, mode='same', axes=(0,1))
    return out/np.amax(out)

# Commands used to make figures or analyze things for the testing and output 
def imshowSave(img, savepath = [], color=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if color:
        ax.imshow(img)
    else:
        ax.imshow(img,cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if savepath:
            plt.savefig(savepath)
            plt.close()
    else:
        plt.close()
    return

# function to plot images in batch and save them to a specified path
def batchPlotImages(batchedColored, batchedSketchified, savepath=[], numplot=5):

    pickIdx = random.sample(range(0, batchedColored.shape[0]), numplot)

    for i, idx in enumerate(pickIdx):
        fig = plt.figure(figsize=(8, 8))
        axisList = addAxis(fig, 1, 2)
        axisList[0].imshow(batchedColored[idx])
        axisList[1].imshow(batchedSketchified[idx], cmap='gray')
        axisList[1].set_title('Complexity: ' +
                              str(complexityMetric(batchedSketchified[idx])))
        groupFormat(axisList)
        plt.tight_layout
        if savepath:
            plt.savefig(savepath + str(i) + '.png')
            plt.close()
        
        #if not savepath:
        #    plt.show()
    return

# Saves colored, sketchified, and color cues images to separate files
def batchPlotImages2(batchedColored, batchedSketchified, batchedColorCues, savepath=[], numplot=5):
    pickIdx = random.sample(range(0, batchedColored.shape[0]), numplot)
    for i, idx in enumerate(pickIdx):
        imshowSave(batchedColored[idx], savepath+str(i)+'_colored.png', color=True)
        imshowSave(batchedSketchified[idx], savepath+str(i)+'_sketch.png', color=False)
        imshowSave(batchedColorCues[idx], savepath+str(i)+'_Cues.png', color=True)

    return

# Generates and saves sketchified images with different block sizes to visualize the effect of block size on sketch quality
def compareSketchifyBlockSize():
    ## This code following was written to show the sketchify process
    ## and make figures for the paper
    data = 'Images_magic_solo/5.jpg'
    colored = read_and_reshape(data)
    savepath = 'SavedImagesForPaper/CompareSketchifyBlockSize/'
    sketchMetric = np.arange(3, 15, 2)
    for blocksize in sketchMetric:
        sketchified = cvSketchify(colored, blocksize)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(sketchified, cmap='gray')
        ax.set_title('blockSize: {} complexity Percent: {}'.format(
            blocksize, int(100 * complexityMetric(sketchified))))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(savepath + str(blocksize) + '.png')
        plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(colored)
    ax.set_title('Original Colored (Downsampled + Cropped)')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(savepath + 'OriginalColor1.png')
    plt.close()
    return

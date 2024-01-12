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
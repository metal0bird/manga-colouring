# Import Statements
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from IPython import display
import datetime
import pickle
import shutil


#creates a 2D grid of subplots within a specified figure (thisfig)
#It takes three parameters: the figure, the number of rows (n1), and the number of columns (n2)
#The function returns a 2D numpy array containing the handles of the created subplots
def addAxis(thisfig, n1, n2):
    axlist = []
    for i in range(n1 * n2):
        axlist.append(thisfig.add_subplot(n1, n2, i + 1))
    return np.array(axlist)

# Hides all ticks and labels on multiple axes to create a cleaner visual display
def groupFormat(axisList):
    for ax in axisList:
        ax.set_xticks([])
        ax.set_yticks([])

# Ensures a clean folder by deleting and recreating it
def delFold_RemakeFold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return

# Sets up a model structure for colorizing images, ready for training
class Color():
    def __init__(self, useColorCues=False, imgsize=256, batchsize=4, 
    testmultiple=5, num_epoch=30, dataSetSize=5580):

        # Define a class which holds the model
        # input image should be square
        self.useColorCues = useColorCues
        self.saveImg = './TrainingSession/'
        self.num_epoch = num_epoch
        self.batch_size = batchsize
        self.number_test = testmultiple*self.batch_size
        self.trainpath = [
            ['Images_holding_staff_CollectionProcessed.pickle'],
            ['Images_solo_magic_CollectionProcessed.pickle'],
            ['Images_forgottagbutlikemagicsolo_CollectionProcessed.pickle'],
            ['Images_tags_girl_solo_CollectionProcessed.pickle']
        ]
        self.testpath = [
            ['TestImages_CollectionProcessed.pickle']
        ]
        self.dataSetSize = dataSetSize # dataSetSize=0 means us max data
        self.traintestSeed = 40

        self.image_size = imgsize
        self.output_size = imgsize

        # define color dimensions
        self.input_colors1 = 1 #greyscale
        self.input_colors2 = 3 #coloured
        self.output_colors = 3

        # define number of units in first hidden layer: generator and discriminator
        self.gf_dim = 64
        self.df_dim = 64
        self.g_filter = 4
        self.d_filter = 4

        # Define generator loss scaling function 
        self.l1_scaling = 100

        # line_images: Sketchified; color_images = color cues; real_images = True colored image
        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors1])
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_colors2])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_colors])

        # Define the Generator
        # Receives both sketched and color cue images as input, combines them, and outputs a predicted colorized image.
        self.combined_preimage = tf.concat([self.line_images, self.color_images],3) # shape batch_size x imgsize x imgsize x 4
        self.generator = self.Generator()
        #tf.keras.utils.plot_model(self.generator, to_file='GeneratorModel.png', show_shapes=True)

        # Define the Discriminator
        # Takes an image (either real or generated) as input and tries to classify it as real or fake
        self.discriminator = self.Discriminator()
        #tf.keras.utils.plot_model(self.discriminator, to_file='DiscriminatorModel.png', show_shapes=True)

        # Define the Optimizer
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.gen_output, self.g_loss, self.d_loss = self.train_step()


    
    # Generator Build Helpers
    # Creates a downsampling block for reducing image size in the generator
    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',    # strides (2 for downsampling)
            kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        # Adds a Leaky ReLU activation layer for non-linearity
        result.add(tf.keras.layers.LeakyReLU())

        return result

    # Creates an upsampling block for increasing image size in the generator
    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,   # strides (2 for upsampling)
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
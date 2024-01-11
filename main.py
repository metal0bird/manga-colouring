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
    

        # Builds the generator model with a U-Net architecture for colorizing images
    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[
            self.image_size, 
            self.image_size,
            self.input_colors1 + self.input_colors2])

        # Defines a list of downsampling blocks with varying filter counts 
        down_stack = [
            self.downsample(self.gf_dim, self.g_filter, apply_batchnorm=False), # (bs, 128, 128, 64)
            self.downsample(self.gf_dim*2, self.g_filter), # (bs, 64, 64, 128)
            self.downsample(self.gf_dim*4, self.g_filter), # (bs, 32, 32, 256)
            self.downsample(self.gf_dim*8, self.g_filter), # (bs, 16, 16, 512)
            self.downsample(self.gf_dim*8, self.g_filter), # (bs, 8, 8, 512)
            self.downsample(self.gf_dim*8, self.g_filter), # (bs, 4, 4, 512)
            self.downsample(self.gf_dim*8, self.g_filter), # (bs, 2, 2, 512)
            self.downsample(self.gf_dim*8, self.g_filter), # (bs, 1, 1, 512)
        ]

        # Defines a list of upsampling blocks with dropout for regularization
        up_stack = [
            self.upsample(self.gf_dim*8, self.g_filter, apply_dropout=True), # (bs, 2, 2, 1024)
            self.upsample(self.gf_dim*8, self.g_filter, apply_dropout=True), # (bs, 4, 4, 1024)
            self.upsample(self.gf_dim*8, self.g_filter, apply_dropout=True), # (bs, 8, 8, 1024)
            self.upsample(self.gf_dim*8, self.g_filter), # (bs, 16, 16, 1024)
            self.upsample(self.gf_dim*4, self.g_filter), # (bs, 32, 32, 512)
            self.upsample(self.gf_dim*2, self.g_filter), # (bs, 64, 64, 256)
            self.upsample(self.gf_dim*1, self.g_filter), # (bs, 128, 128, 128)
        ]

        # Creates the final layer using a transposed convolutional layer with output_colors filters and a tanh activation for output images
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_colors, self.g_filter,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh') # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        # Applies the final layer to produce the colorized output image.
        x = last(x)

        # Returns a Keras model with the defined inputs and outputs.
        return tf.keras.Model(inputs=inputs, outputs=x)

    # Calculates losses to guide the generator towards realistic and accurate colorization
    def generator_loss(self, disc_generated_output, gen_output, target):
        # Uses binary cross-entropy loss to measure how well the discriminator was fooled by generated images
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.l1_scaling * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    # Discriminator Build Helpers
    # Builds the discriminator model to classify images as real or generated
    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[self.image_size, self.image_size, self.input_colors1 + self.input_colors2], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.image_size, self.image_size, self.output_colors], name='target_image')

        # Both images are first concatenated, creating a single input of (batch_size, image_size, image_size, input_colors1 + input_colors2) channels
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

        # the discriminator uses a series of downsampling blocks to progressively reduce the image size and extract higher-level features
        # A convolutional layer with increasing filter counts (64, 128, 256) to capture intricate details
        down1 = self.downsample(self.df_dim, self.d_filter, False)(x) # (bs, 128, 128, 64)
        down2 = self.downsample(self.df_dim*2, self.d_filter)(down1) # (bs, 64, 64, 128)
        down3 = self.downsample(self.df_dim*4, self.d_filter)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(self.df_dim*8, self.d_filter, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        # This layer has a single filter and serves as the final decision-making step
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
        
        # The output of the discriminator is a single value for each image in the batch, representing the probability of that image being a real image
        # A value close to 1 indicates a high probability of being real, while a value close to 0 indicates a high probability of being fake (generated)
        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    # Calculates and combines losses to train the discriminator
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    
    # Otpimizer Step
    # Performs a training step, updating generator and discriminator
    @tf.function
    def train_step(self):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(self.combined_preimage, training=True)
            disc_real_output = self.discriminator([self.combined_preimage, self.real_images], training=True)
            disc_generated_output = self.discriminator([self.combined_preimage, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, self.real_images)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))

        return gen_output, gen_total_loss, disc_loss
    

    ## Class Functions Calls 
    # Initializes the model and attempts to load a checkpoint
    def loadModel(self):
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()
    
        if self.load("./checkpoint"):
            print("Loaded")
        else:
            print("Load failed")
        return

    # Loads a model checkpoint from a specified directory
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
        
    # to save the model at a directory at a checkpoint
    def save(self, checkpoint_dir, step):
        model_name = "model"
        model_dir = "tr"
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
    
    # Shuffles three arrays in unison, maintaining their corresponding elements
    # - Ensures all arrays have the same length
    # - Creates a random permutation of indices using a fixed seed for reproducibility
    # - Returns shuffled copies of the arrays
    def unison_shuffled_copies(self, a, b, c):
        assert a.shape[0] == b.shape[0] == c.shape[0]
        p = np.random.RandomState(seed=self.traintestSeed).permutation(len(a))
        return a[p], b[p], c[p]

    # Loads and unpacks data from multiple pickle files
    # - Creates empty arrays to hold colored images, color cues, and sketchified images
    # - Iterates through each pickle file:
    #   - Loads data from the file
    #   - Extracts colored images, sketchified images, and color cues (if enabled)
    #   - Concatenates extracted data to the respective arrays
    # - Removes the initial zero element from each array
    # - Normalizes colored images to the range [0, 1]
    # - Returns the unpacked arrays

    def unpackProcessedPickle(self, usepath):
        # create holder since usepath can have multiple pickle files
        coloredImages = np.zeros((1,256,256,3), dtype=float)
        colorCues = np.zeros((1,256,256,3), dtype=float)
        sketchifiedImages = np.zeros((1,256,256,1), dtype=float)
        for path in usepath:
            with open(path[0], 'rb') as f:
                loadeddata = pickle.load(f)
                coloredImages_ = loadeddata['batchedColored']
                sketchifiedImages_ = np.expand_dims(loadeddata['batchedSketchified'],3)  
                # Select Color Cues 
                if not self.useColorCues: #use Cues are False
                    colorCues_ = np.ones_like(coloredImages_)
                elif self.useColorCues:
                    colorCues_ = loadeddata['batchedColorCues']

                coloredImages = np.concatenate((coloredImages, coloredImages_),0)
                colorCues = np.concatenate((colorCues, colorCues_),0)
                sketchifiedImages = np.concatenate((sketchifiedImages, sketchifiedImages_),0)
        
        # Remove First zero element and Normalize as needed:
        coloredImages = coloredImages[1:]/255.0
        colorCues = colorCues[1:]
        sketchifiedImages = sketchifiedImages[1:]
        return coloredImages, colorCues, sketchifiedImages


    # Loads and prepares training and test data from pickle files
    def getModelTrainData(self):
        coloredImages, colorCues, sketchifiedImages = self.unpackProcessedPickle(usepath=self.trainpath)

        if self.dataSetSize == 0:
            self.dataSetSize = coloredImages.shape[0]

        # reduce number images in the loaded data if desired:
        coloredImages = coloredImages[:self.dataSetSize]
        colorCues = colorCues[:self.dataSetSize]
        sketchifiedImages = sketchifiedImages[:self.dataSetSize]

        # shuffle then set aside a grouping for testing after the training
        coloredImages, colorCues, sketchifiedImages = self.unison_shuffled_copies(
            coloredImages, colorCues, sketchifiedImages)

        testColored = coloredImages[:self.number_test]
        testSketch = sketchifiedImages[:self.number_test]
        testColorCues = colorCues[:self.number_test]
        coloredImages = coloredImages[self.number_test:]
        sketchifiedImages = sketchifiedImages[self.number_test:]
        colorCues = colorCues[self.number_test:]

        print(
            'Training Data Shape:\n Colored: {}\n Sketches: {}\n ColorCues: {} '.format(
                coloredImages.shape, sketchifiedImages.shape, colorCues.shape)
        )
        print(
            'Check Norm via Max Values:\n Colored: {}\n Sketches: {}\n ColorCues: {} '.format(
                np.amax(coloredImages), np.amax(sketchifiedImages), np.amax(colorCues))
        )
        return testColored, testSketch, testColorCues, coloredImages, sketchifiedImages, colorCues
        
    # Generates and saves images from test data using the trained model
    def generate_images(self, testColored, testSketch, testColorCues, saveFolder):
        imgcounter = 0
        savepath = self.saveImg + saveFolder +'/'
        delFold_RemakeFold(savepath)
        number_test_here = testColored.shape[0]
        for teststep in range(number_test_here//self.batch_size):
            #iterate to get a batch
            checkReal =  testColored[teststep*self.batch_size:(teststep+1)*self.batch_size]
            checkSketch = testSketch[teststep*self.batch_size:(teststep+1)*self.batch_size]
            checkColor =  testColorCues[teststep*self.batch_size:(teststep+1)*self.batch_size]
            
            evaluatedImages = self.sess.run(
                        [self.gen_output], 
                        feed_dict={
                            self.real_images: checkReal,
                            self.line_images: checkSketch,
                            self.color_images: checkColor
                            }
                        )
            evaluatedImages = np.squeeze(np.array(evaluatedImages))
            print(evaluatedImages.shape)
            for idx in range(self.batch_size):
                # iterate over each image in the batch
                savehere = savepath +str(imgcounter)+'.png'
                self.viewTestResults(
                    savehere,
                    np.squeeze(checkReal[idx]), np.squeeze(checkSketch[idx]),
                    np.squeeze(checkColor[idx]), np.squeeze(evaluatedImages[idx])
                )   
                imgcounter+=1
        return
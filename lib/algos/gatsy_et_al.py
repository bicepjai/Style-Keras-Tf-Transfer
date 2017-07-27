"""
code used from https://github.com/chiphuyen/tf-stanford-tutorials

"""

from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from lib.models.vgg_models import load_vgg19
from utils import lazy_property, get_resized_image, generate_noise_image, save_image

from config import PRETRAINED_MODEL_LIST, STYLE_ALGO_LIST, CONTENT_STYLE_LAYERS_WEIGHTS

import logging
logger = logging.getLogger('style_transfer')


"""
MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering.
The input images should be zero-centered by mean pixel (rather than mean image) subtraction.
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))


class StyleTransfer1:
    """
    This class creates style transfer based on paper
    "A Neural Algorithm of Artistic Style" by Gatys et al.
    """
    def __init__(self, parameters):

        self.params = parameters
        # training parameters
        # epochs, learning_rate, batch_size, gpu_device_id, log_dir

        # styling parameters
        # style_image, content_image, pretrained_model
        # image_height, image_width, noise_ratio
        # content_weight, style_weight

        # dictinaries with { layer_name : weight } as values
        self.content_layers = CONTENT_STYLE_LAYERS_WEIGHTS[self.params.pretrained_model]['content']
        self.style_layers = CONTENT_STYLE_LAYERS_WEIGHTS[self.params.pretrained_model]['style']
        logger.debug("content_layers: %s",str(self.content_layers))
        logger.debug("style_layers: %s",str(self.style_layers))


        self.content_image = None
        self.style_image = None

        # global variable that tracks epochs executed across runs
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="global_step")
        self.model = None

        # class private methods that can be called only once
        self._image_processing
        self._config_input
        self._config_loss
        self._config_summaries
        self._config_optimizer


    @lazy_property
    def _image_processing(self):
        """
        prepare images before starting to train
        """
        logger.debug("setting up tf input pipeline ...")

        # image processing
        content_image = get_resized_image(self.params.content_image, self.params.image_height, self.params.image_width)
        self.content_image = content_image - MEAN_PIXELS
        style_image = get_resized_image(self.params.style_image, self.params.image_height, self.params.image_width)
        self.style_image = style_image - MEAN_PIXELS

        # this non tf variable will hold the final generated image
        self.initial_image = generate_noise_image(self.content_image, \
                                        self.params.image_height, self.params.image_width, \
                                        self.params.noise_ratio)

    @lazy_property
    def _config_input(self):
        """
        Input image feed through pipeline setup
        """
        logger.debug("setting up tf input pipeline ...")

        with tf.variable_scope('input') as scope:
            # use variable instead of placeholder because we're training the intial image to make it
            # look like both the content image and the style image
            self.input_image = tf.Variable(np.zeros([1, self.params.image_height, self.params.image_width, 3]),
                                                dtype=tf.float32)

            if self.params.pretrained_model == 'vgg19':
                self.model = load_vgg19(self.input_image)
            else:
                raise Exception("Model cannot be not used for training style \
                                 tranfer in StyleTransfer1 class")

    def _create_content_loss(self, p, f):
        """
        Calculate the loss between the feature representation of the
        content image and the generated image.

        Inputs:
            p, f are just P, F in the paper
            (read the assignment handout if you're confused)
            Note: we won't use the coefficient 0.5 as defined in the paper
            but the coefficient as defined in the assignment handout.
        Output:
            the content loss

        """
        logger.debug("setting up tf content loss ...")
        with tf.name_scope('content_loss') as scope:
            content_loss = tf.reduce_sum(tf.square(p - f)) / (4 * p.size)
            return content_loss

    def _gram_matrix(self, F, N, M):
        """
        Create and return the gram matrix for tensor F
        """
        F = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(F), F)

    def _single_style_loss(self, a, g):
        """ Calculate the style loss at a certain layer
        Inputs:
            a is the feature representation of the real image
            g is the feature representation of the generated image
        Output:
            the style loss at a certain layer (which is E_l in the paper)

            1. using the function _gram_matrix()
            2. using the same coefficient for style loss as in the paper
            3. a and g are feature representation, not gram matrices
        """
        N = a.shape[3] # filters
        M = a.shape[1] * a.shape[2] # height x width
        A = self._gram_matrix(a, N, M)
        G = self._gram_matrix(g, N, M)
        return tf.reduce_sum(tf.square(G - A)) / (2 * N * N * M * M)

    def _create_style_loss(self, A):
        """
        Return the total style loss
        A is a dictionary with style layer names as keys
        """
        logger.debug("setting up tf style loss ...")
        with tf.variable_scope('style_loss') as scope:
            style_loss = 0
            for style_layer in self.style_layers:
                style_loss += self.style_layers[style_layer] * (self._single_style_loss(A[style_layer], self.model[style_layer]))

            return style_loss

    @lazy_property
    def _config_loss(self):
        logger.debug("gathering style and content loss ...")
        with tf.variable_scope('loss') as scope:
            # content loss
            with tf.Session() as sess:
                sess.run(self.input_image.assign(self.content_image))
                # since the content layer is just one, we are using this
                # need to be updated when content layer is not just one
                content_layer =  list(self.content_layers)[0]
                p = sess.run(self.model[content_layer])

            content_loss = self._create_content_loss(p, self.model[content_layer])

            # style loss
            with tf.Session() as sess:
                sess.run(self.input_image.assign(self.style_image))
                A = {}
                for layer_name in self.style_layers:
                    A[layer_name] = sess.run(self.model[layer_name])

            style_loss = self._create_style_loss(A)

            # total loss
            total_loss = self.params.style_weight * style_loss + self.params.content_weight * content_loss

        self.model['content_loss'] = content_loss
        self.model['style_loss']   = style_loss
        self.model['total_loss']   = total_loss

    @lazy_property
    def _config_summaries(self):
        """
        Create summary ops necessary
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar("total_loss",self.model['total_loss'])
            tf.summary.histogram("total_loss",self.model['total_loss'])
            tf.summary.scalar("content_loss",self.model['content_loss'])
            tf.summary.histogram("content_loss",self.model['content_loss'])
            tf.summary.scalar("style_loss",self.model['style_loss'])
            tf.summary.histogram("style_loss",self.model['style_loss'])
            self.model['summary_op'] = tf.summary.merge_all()

    @lazy_property
    def _config_optimizer(self):
        self.model['optimizer'] = tf.train.AdamOptimizer(self.params.learning_rate) \
                                    .minimize(self.model['total_loss'], \
                                              global_step = self.global_step)

    def train(self):
        """
        Train your model.
        folders for checkpoints and outputs will be created in the log directory
        """
        # https://stackoverflow.com/questions/37337728/tensorflow-internalerror-blas-sgemm-launch-failed
        if 'session' in locals() and session is not None:
            logger.debug('Close interactive session')
            session.close()

        # processed image or images after provided number of epochs will be stored here
        logging.debug("creating output folder for stylsed images to be stored %s" % (self.params.log_dir+'/outputs'))
        if not os.path.exists(self.params.log_dir+'/outputs'):
            os.makedirs(self.params.log_dir+'/outputs')

        logger.info("Training ... total epochs:%d" % (self.params.epochs))
        skip_step = 1
        # with tf.device("/gpu:1"):
        #     config_sp = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session() as sess:
            saver = tf.train.Saver()

            ## initialize your variables
            sess.run(tf.global_variables_initializer())

            ## create writer to write your graph
            writer = tf.summary.FileWriter(self.params.log_dir+'/graphs',sess.graph)

            # assigning initial_image to the input_image for training
            sess.run(self.input_image.assign(self.initial_image))

            # loading checkpoints if available
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.params.log_dir+'/checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = self.global_step.eval()

            logger.debug("total_epochs: %d",self.params.epochs)
            logger.debug("current_initial_step: %d",initial_step)

            # for measuring time when required
            start_time = time.time()

            # training
            for epoch in tqdm(range(initial_step, self.params.epochs)):
                if epoch >= 5 and epoch < 20:
                    skip_step = 10
                elif epoch >= 20:
                    skip_step = self.params.checkpoint_epochs

                sess.run(self.model['optimizer'])

                # gathering processed images
                if (epoch + 1) % skip_step == 0:
                    # obtain generated image and loss
                    gen_image, total_loss, summary = sess.run([self.input_image,
                                                        self.model['total_loss'], self.model['summary_op']])
                    # refer the comments at the start of this file for reference
                    gen_image = gen_image + MEAN_PIXELS

                    filename = self.params.log_dir+'/outputs/%d.png' % (epoch)
                    save_image(filename, gen_image)

                # gathering summaries for tensorboard
                if (epoch + 1) % self.params.summary_epochs == 0:
                    writer.add_summary(summary, global_step=epoch)
                    # logger.debug('Step {}\n   Sum: {:5.1f}'.format(epoch + 1, np.sum(gen_image)))
                    # logger.debug('   Loss: {:5.1f}'.format(total_loss))
                    # logger.debug('   Time: {}'.format(time.time() - start_time))
                    start_time = time.time()

                # checkpointing model params
                if (epoch + 1) % self.params.store_image_epochs == 0:
                    saver.save(sess, self.params.log_dir+'/checkpoints/style_transfer', epoch)

            writer.close()
            logger.info("Training Done ! look into %s folder for stylised images" % (self.params.log_dir+'/outputs'))

"""
code used from https://github.com/chiphuyen/tf-stanford-tutorials

"""
from __future__ import print_function

import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib
import functools

import logging
logger = logging.getLogger('style_transfer')


# http://danijar.com/structuring-your-tensorflow-models/
def lazy_property(function):
    """
    Annotation for methods in a class that make sure it executes only once
    """
    attribute = '_cache_' + function.__name__
    logger.info(attribute)
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def download_pretrained_model(download_link, file_name, expected_bytes):
    """
    Download the pretrained model if it's not already downloaded
    """

    if os.path.exists(file_name):
        logger.debug("Pretrained model already downloaded")
        return

    logger.info("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        logger.debug('Successfully downloaded the file', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. try downloading from a browser.')

def get_resized_image(img_path, height, width, save=True):
    """
    self explanatory
    """
    image = Image.open(img_path)
    # because PIL is column major so you have to change place of width & height
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)


def generate_noise_image(content_image, height, width, noise_ratio=0.6):
    """
    self explanatory
    """
    noise_image = np.random.uniform(-20, 20,
                                    (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)


def save_image(path, image):
    """
    self explanatory
    """
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def get_image_dims(img_path):
    """
    returns image dimensions
    """
    image = Image.open(img_path)
    return image.size

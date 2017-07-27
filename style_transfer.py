
"""
style-tf-transfer.py

This file is the entry point for the execution of the algorithms what gathers commandline options
and validation are performed before being used to perform the task of stryle transfer.
"""

__author__ = "Jayaram Prabhu Durairaj"
__credits__ = ["Jayaram Prabhu Durairaj"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jayaram Prabhu Durairaj"
__email__ = "bicepjai@gmail.com"

from __future__ import print_function
from argparse import ArgumentParser
import os, sys, shutil
from time import gmtime, strftime
import logging
import tensorflow as tf

# path setup to access lib folder
sys.path.insert(0, 'lib')

# local imports
from setup import style_it
from utils import get_image_dims
from config import PRETRAINED_MODEL_LIST, STYLE_ALGO_LIST, CONTENT_STYLE_LAYERS_WEIGHTS

# default values
from config import STYLE_IMAGE, STYLE_ALGO, PRETRAINED_MODEL
from config import LOW_QUAL_IMAGE_DIMS, NOISE_RATIO, CONTENT_WEIGHT, STYLE_WEIGHT
from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, GPU_DEVICE_ID
from config import STORE_EVERY_EPOCHS, CHECKPOINT_EVERY_EPOCHS, SUMMARY_EVERY_EPOCHS

def build_parser():
    """
    Returns parser that contains command line arguments
    """
    parser = ArgumentParser()

    # required arguments
    parser.add_argument('--content_image', type=str, dest='content_image',
                        help='image on which the style will be transfered',
                        metavar='CONTENT_IMAGE', required=True)

    # arguments that can be set a default values
    parser.add_argument('--style_image', type=str, dest='style_image',
                        help='image from which the style will be gathered, default starry_night',
                        metavar='STYLE_IMAGE', default=STYLE_IMAGE)

    parser.add_argument('--style_algo', type=str, dest='style_algo',
                        help='style transfer algorithm used',
                        metavar='STYLE_ALGO', default=STYLE_ALGO,
                        choices=STYLE_ALGO_LIST)

    parser.add_argument('--pretrained_model', type=str, dest='pretrained_model',
                        help='pretrained model used for style transfer',
                        metavar='PRETRAINED_MODEL', default=PRETRAINED_MODEL,
                        choices=PRETRAINED_MODEL_LIST)

    parser.add_argument('--image_dims', type=str, dest='image_dims', nargs=2,
                        help='content image dimensions considered for style transfer <<height> <width>>',
                        metavar='INT_VALUE', default=None)

    parser.add_argument('--noise_ratio', type=float, dest='noise_ratio',
                        help='percentage of weight of the noise for intermixing with the content image',
                        metavar='NOISE_RATIO', default=NOISE_RATIO)

    parser.add_argument('--content_weight', type=float, dest='content_weight',
                        help='the weight of content layer loss in total loss',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style_weight', type=float, dest='style_weight',
                        help='the weight of style layer loss in total loss',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument("-v", '--verbose', dest='verbose',
                        help='logging verbose information',
                        action="store_true")

    # tensorflow arguments
    parser.add_argument('--learning_rate', type=float, dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--epochs', type=float, dest='epochs',
                        help='training epochs',
                        metavar='EPOCHS', default=EPOCHS)

    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        help='batch size for eval',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--gpu_device_id', type=int, dest='gpu_device_id',
                        help='device for eval. CPU discouraged. ex: 0',
                        metavar='DEVICE', default=GPU_DEVICE_ID)

    parser.add_argument('--enable_tf_logs', dest='enable_tf_logs',
                        help='disabling tf log messages',
                        action="store_true")

    # logging parameters

    parser.add_argument('--dont_log', dest='dont_log',
                        help='keel log files and dir at end of processing',
                        action="store_true")

    parser.add_argument('--store_image_epochs', type=float, dest='store_image_epochs',
                        help='after these many epochs images will be stored in a folder names output inside log directory',
                        metavar='STORE_EVERY_EPOCHS', default=STORE_EVERY_EPOCHS)

    parser.add_argument('--checkpoint_epochs', type=float, dest='checkpoint_epochs',
                        help='checkpoints during training must be logged every these many epochs',
                        metavar='STORE_EVERY_EPOCHS', default=CHECKPOINT_EVERY_EPOCHS)

    parser.add_argument('--summary_epochs', type=float, dest='summary_epochs',
                        help='gathering summaries every these many epochs',
                        metavar='SUMMARY_EVERY_EPOCHS', default=SUMMARY_EVERY_EPOCHS)

    return parser


def validate_arguments(options, parser):
    """
    Validating passed in command line arguments
    """
    if not os.path.exists(options.content_image):
        logging.error("Please provide valid content image path !")
        parser.print_help()
        exit(1)

    if not os.path.exists(options.style_image):
        logging.error("Please provide valid style image path !")
        parser.print_help()
        exit(1)

    # validating image dimensions
    if options.image_dims == None:
        (options.image_height, options.image_width) = get_image_dims(options.content_image)
    else:
        try:
            image_dims = tuple(map(int, options.image_dims))
            print (image_dims)
            (options.image_height, options.image_width) = image_dims
            logging.debug("content image dimensions (%d x %d)" % (options.image_height, options.image_width))
        except ValueError as e:
            logging.error("Please provide proper image dimensions !")
            parser.print_help()
            exit(1)

    # validating logs
    if not options.dont_log:
        style_name = os.path.splitext(os.path.basename(options.style_image))[0]
        content_image = os.path.splitext(os.path.basename(options.content_image))[0]

        options.log_dir = "tmp_dir_"
        options.log_dir += style_name + "_" + content_image + "_"
        options.log_dir += str(options.image_height) + "_" + str(options.image_width)

        logging.debug("creating logging folder %s" % options.log_dir)
        if not os.path.exists(options.log_dir):
            os.makedirs(options.log_dir)
            assert os.path.exists(options.log_dir), "log dir not formed!"

    if not options.enable_tf_logs:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    assert options.learning_rate >= 0
    assert options.epochs > 0
    assert options.batch_size > 0


def main():
    """
    This is where it all begins :)
    """

    parser = build_parser()
    options = parser.parse_args()

    # logging setup
    LOG_FORMAT = "%(asctime)s.%(msecs)03d:%(filename)s:%(funcName)s :: %(message)s"
    level = logging.DEBUG if options.verbose else logging.INFO
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logger = logging.getLogger('style_transfer')
    validate_arguments(options, parser)

    logger.info("Starting style transfer...")
    logger.debug("options provided : \n%s" % str(options))
    style_it(options)


if __name__ == '__main__':

    main()

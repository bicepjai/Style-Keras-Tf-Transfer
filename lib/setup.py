
import os, sys
import logging
from algos.gatsy_et_al import StyleTransfer1

import logging
logger = logging.getLogger('style_transfer')

def style_it(parameters):
    """
    style transfer algorithm choosen based on arguments
    the parameters passed in are argparse options namespace object
    they can be accessed by using <parameters>.<property_name>
    """

    if parameters.style_algo == 'gatsy_et_al':
        model = StyleTransfer1(parameters)
        model.train()

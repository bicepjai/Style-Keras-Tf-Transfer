
"""
config.py
This file contains configurations used to style transfer algorithms and the available options
that can be used in this project. Command line option validation are also performed on choices
available from this configuration file.
"""

__author__ = "Jayaram Prabhu Durairaj"
__credits__ = ["Jayaram Prabhu Durairaj"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Jayaram Prabhu Durairaj"
__email__ = "bicepjai@gmail.com"


PRETRAINED_MODEL_LIST = ['vgg19']

STYLE_ALGO_LIST = [
                    'gatsy_et_al', # StyleTransfer1
                  ]

# Layers used for styling features
# usually more weights given for deeper layers
CONTENT_STYLE_LAYERS_WEIGHTS = {
            "vgg19"  : {
                             "content"  :   {
                                                "conv4_2": 1
                                            },
                             "style"    :   {
                                                "conv1_1": 0.5,
                                                "conv2_1": 1.0,
                                                "conv3_1": 1.5,
                                                "conv4_1": 3.0,
                                                "conv5_1": 4.0
                                            }
                        },

            "googlenet" : {
                            "content": {
                                            "conv2/3x3": 2e-4,
                                            "inception_3a/output": 1-2e-4
                                        },
                            "style":
                                        {
                                            "conv1/7x7_s2": 0.2,
                                            "conv2/3x3": 0.2,
                                            "inception_3a/output": 0.2,
                                            "inception_4a/output": 0.2,
                                            "inception_5a/output": 0.2
                                        }
                            }
        }

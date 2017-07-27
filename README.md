# style-transfer

## Introduction

This project is intented to create all the existing style transfer methods in tensorflow usiong proper
software development methodologies and appropriate comments  in necessary places. This is an attempt to
break away from hacked-up scripts that has no comments and design.

Style transfers available:

1. A Neural Algorithm of Artistic Style" by L. Gatys, A. Ecker, and M. Bethge
   paper: http://arxiv.org/abs/1508.06576. 

2. ...

## Requirements

 - Python >= 2.7
 - CUDA >= 6.5 (highly recommended)
 - Tensorflow
 - Image

It is suggested to use GPU fro high quality pictures

## Usage
```
python style_transfer.py -h

usage: style_transfer.py [-h] --content_image CONTENT_IMAGE
                         [--style_image STYLE_IMAGE] [--style_algo STYLE_ALGO]
                         [--pretrained_model PRETRAINED_MODEL]
                         [--image_dims INT_VALUE INT_VALUE]
                         [--noise_ratio NOISE_RATIO]
                         [--content_weight CONTENT_WEIGHT]
                         [--style_weight STYLE_WEIGHT] [-v]
                         [--learning_rate LEARNING_RATE] [--epochs EPOCHS]
                         [--batch_size BATCH_SIZE] [--gpu_device_id DEVICE]
                         [--enable_tf_logs] [--dont_log]
                         [--store_image_epochs STORE_EVERY_EPOCHS]
                         [--checkpoint_epochs STORE_EVERY_EPOCHS]
                         [--summary_epochs SUMMARY_EVERY_EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --content_image CONTENT_IMAGE
                        image on which the style will be transfered
  --style_image STYLE_IMAGE
                        image from which the style will be gathered, default
                        starry_night
  --style_algo STYLE_ALGO
                        style transfer algorithm used
  --pretrained_model PRETRAINED_MODEL
                        pretrained model used for style transfer
  --image_dims INT_VALUE INT_VALUE
                        content image dimensions considered for style transfer
                        <<height> <width>>
  --noise_ratio NOISE_RATIO
                        percentage of weight of the noise for intermixing with
                        the content image
  --content_weight CONTENT_WEIGHT
                        the weight of content layer loss in total loss
  --style_weight STYLE_WEIGHT
                        the weight of style layer loss in total loss
  -v, --verbose         logging verbose information
  --learning_rate LEARNING_RATE
                        learning rate (default 1)
  --epochs EPOCHS       training epochs
  --batch_size BATCH_SIZE
                        batch size for eval
  --gpu_device_id DEVICE
                        device for eval. CPU discouraged. ex: 0
  --enable_tf_logs      disabling tf log messages
  --dont_log            keel log files and dir at end of processing
  --store_image_epochs STORE_EVERY_EPOCHS
                        after these many epochs images will be stored in a
                        folder names output inside log directory
  --checkpoint_epochs STORE_EVERY_EPOCHS
                        checkpoints during training must be logged every these
                        many epochs
  --summary_epochs SUMMARY_EVERY_EPOCHS
                        gathering summaries every these many epochs
```


## Reference
1. This is an extension from homework assignments from cs20si
   https://github.com/chiphuyen/tf-stanford-tutorials
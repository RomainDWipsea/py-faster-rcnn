# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
#from datasets.basketball import basketball
#from datasets.example import example
from datasets.turtle import turtle

import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up basketball_<split>
basketball_path = '/share/cluster/deeplearning/FRCNN_Romain/data/basketball/'
for split in ['train', 'test']:
    name = '{}_{}'.format('basketball', split)
    __sets[name] = (lambda split=split: basketball(split, basketball_path))

# Set up turtle_<split>
turtle_path = '/home/romain/ext/py-faster-rcnn/data/turtle/'
for split in ['train', 'test']:
    name = '{}_{}'.format('turtle', split)
    __sets[name] = (lambda split=split: turtle(split, turtle_path))

#Set path, and name of dataset
example_dataset_path = '/share/cluster/deeplearning/FRCNN_Romain/data/example'
for split in ['train', 'test']:
    name = '{}_{}'.format('example', split)
    __sets[name] = (lambda split=split: example(split, example_dataset_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

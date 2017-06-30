#!/usr/bin/env python
# --------------------------------------------------------
# Fast R-CNN Application to Wipsea data
# Copyright (c) 2017 Wipsea
#  Written by Romain Dambreville
# --------------------------------------------------------

"""Use a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import use_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
#from datasets.factory import get_imdb
from datasets.turtle import turtle
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Pase input arguments
    """
    parser = argparse.ArgumentParser(description='Use a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--path', dest='turtle_path',
                        help='folder of images to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    else:
        sys.exit(1)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    
    for split in ['train', 'test']:
        name = '{}_{}'.format('turtle', split)
        imdb_data = (lambda split=split: turtle(split, args.turtle_path))

    use_net(net, imdb_data(), max_per_image=args.max_per_image, vis=args.vis)


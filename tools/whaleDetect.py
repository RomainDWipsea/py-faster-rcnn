#!/usr/bin/env python

# --------------------------------------------------------
# Whale detection
# Copyright (c) 2017 Wipsea
# Written by Romain Dambreville
# --------------------------------------------------------

"""
Script taking a folder of images and returning a file containing
all the images with detection > threshold with the corresponding position.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from glob import glob

CLASSES = ('__background__',
           'megaptera')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'zf_faster_rcnn_megaExp1_iter_3000.caffemodel')}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Whale detection script')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--folder', dest='my_folder',
                        help='Source folder of images', default=None, type=str)

    args = parser.parse_args()

    return args

def exportRes(bboxes, im_name, output_file):
    file = open(output_file,"a")
    for line in range(0, len(bboxes)):
        #for s, (a,b,c,d,e) in bboxes[line]:
        file.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(im_name, bboxes[line][0],
                       bboxes[line][1][0], bboxes[line][1][1],
                       bboxes[line][1][2], bboxes[line][1][3]))

        #file.write(im_name + ' ' + str(bboxes[line]) + '\n')
                       #s + a + b + c + d + '\n')
    file.close()

def cutAndDetect(im_name):
    """ Extract sub images to fit the network at a reasonnable size """
    im_file = os.path.join(cfg.DATA_DIR, 'whales', im_name)
    im = cv2.imread(im_file)
    res = []
    for (x, y, window) in slidingWindow(im, step=650, windowSize=(1000,750)):
        if window.shape[0] != 750 or window.shape[1] != 1000:
            continue
        scores, boxes = im_detect(net, window)

        cls_ind = 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.3)
        dets = dets[keep, :]
        for i in range(0, len(scores)):
            if scores[i,1] > 0.1:
                vis_detections(window, 'megaptera', dets,  im_name, 0.1)
                rescaleBox = boxes[i,4:8]
                rescaleBox[0]+= x
                rescaleBox[1]+= y
                rescaleBox[2]+= x
                rescaleBox[3]+= y
                tmp=(scores[i,1],rescaleBox)
                res.append(tmp)
    print res
    return res

def slidingWindow(image, step, windowSize):
    for y in xrange(0, image.shape[0], step):
        for x in xrange(0, image.shape[0], step):
            yield(x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

def vis_detections(im, class_name, dets, image_name, thresh=0.3):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
       fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(class_name+image_name)
    plt.draw()


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'turtle_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n Did you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = glob(os.path.join(args.my_folder, '*.JPG'))
    # im_names = ['DJI_0044_0.JPG', 'DJI_0045_0.JPG', 'DJI_0046_0.JPG', 'DJI_0046_1.JPG']

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        res = cutAndDetect(im_name)
        exportRes(res, im_name, "results.txt");

        #TODO : function to cut the images at the right size and put them in a "vector"
        #for subimages in vector of subimages : whaleDetect
        #TODO : print results in a file

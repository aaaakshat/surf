#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="/dev/video0", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# create video output object
raw_out = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
simple_out = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# define function to determine bit state
def B(k, cutoff):
    if (k >= cutoff):
        return 1
    else:
        return 0

# apply B(k) to raw data to generate data cube
b_vec = np.vectorize(B)

import matplotlib.pyplot as plt

# qis code
def add_noise(frame):
    # extract b/w frame from video stream
    bw = jetson.utils.cudaAllocMapped(width=frame.width, height=frame.height, format="gray8")
    jetson.utils.cudaConvertColor(frame, bw)

    # convert to arr for manipulation
    alpha = 0.01
    T = 10
    q=0.1
    x = jetson.utils.cudaToNumpy(bw)
    y = np.empty([T, int(bw.height/2), int(bw.width/2)])
    x = x[::2,::2]
    x.shape = (y[0].shape)
    print("x shape: ", x.shape)

    for i in range(0, T):
        y[i] = np.random.poisson(lam=alpha*x, size=x.shape)
        y[i] = b_vec(y[i], q)

    y_mean = np.apply_along_axis(np.mean, 0, y)*255
    y_mean = y_mean.astype(int)

    # convert arr back to cudaImage for render
    bw = jetson.utils.cudaAllocMapped(width=y_mean.shape[0], height=y_mean.shape[1], format="gray8")
    bw = jetson.utils.cudaFromNumpy(y_mean)

    plt.imshow(y_mean, cmap="gray")
    plt.show()
    print("ymean shape: ", y_mean.shape)

    a = jetson.utils.cudaToNumpy(bw).T
    plt.imshow(a[0].T, cmap="gray")
    plt.show()

    frame = jetson.utils.cudaAllocMapped(width=bw.width, height=bw.height, format="rgba8")
    jetson.utils.cudaConvertColor(bw, frame)

    print("final output:")
    a = jetson.utils.cudaToNumpy(frame).T
    plt.imshow(a[0].T, cmap="gray")
    plt.show()
    return frame

import time

while True:
# capture the next frame
    print("input capture")
    frame = input.Capture()

# render the frame
    print("raw")
   # raw_out.Render(frame)
    print("adding noise")
    out = add_noise(frame)
    print("rendering")
    a = jetson.utils.cudaToNumpy(out)
    simple_out.Render(out)

    print("waiting")
    time.sleep(1)


# update the title bar
#out1.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
#out2.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

# print out performance info
# net.PrintProfilerTimes()

# exit on input/output EOS
#if not input.IsStreaming() or not output.IsStreaming():
#    break



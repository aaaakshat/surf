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
import random

fw = 640
fh = 360
bw = jetson.utils.cudaAllocMapped(width=2*fw, height=2*fh, format="gray8")
out = jetson.utils.cudaAllocMapped(width=fw, height=fh, format="gray8")
outrgb = jetson.utils.cudaAllocMapped(width=fw, height=fh, format="rgb8")

# qis code
def add_noise(frame):
    # extract b/w frame from video stream
    jetson.utils.cudaConvertColor(frame, bw)

    # convert to arr for manipulation
    alpha = 0.01
    T = 10
    q=0.1
    x = jetson.utils.cudaToNumpy(bw)
    x = x[::2,::2]
#       need to get the luma channel from img
    x_0 = x.T[0].T
    x_0 = np.random.poisson(lam=alpha*x_0, size=x_0.shape)/alpha

    out = jetson.utils.cudaFromNumpy(x_0)
    jetson.utils.cudaConvertColor(out, outrgb)

    return outrgb

while True:
# capture the next frame
    print("input capture")
    frame = input.Capture()

# render the frame
    print("raw")
    raw_out.Render(frame)
    print("adding noise")
    out = add_noise(frame)
    print("rendering")
    simple_out.Render(out)



# update the title bar
#out1.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
#out2.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

# print out performance info
# net.PrintProfilerTimes()

# exit on input/output EOS
#if not input.IsStreaming() or not output.IsStreaming():
#    break



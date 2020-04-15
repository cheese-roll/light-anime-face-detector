"""
    Run detector for each image in the directory.
"""

import os
import cv2
import sys
import glob
import json
import time
import random
import hashlib
import imutils
import argparse
import more_itertools
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from core.detector import LFFDDetector

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run detector on images in directory.")
    parser.add_argument("-i", "--input-directory", help="Input images directory.", required=True)
    parser.add_argument("-d", "--detector-path", help="Detector JSON configuration path.", required=True)
    parser.add_argument("--display-height", help="Display height (default=%(default)s).", type=int, default=720)
    parser.add_argument("--use-gpu", help="Use GPU or not (default=%(default)s).", type=int, default=1)
    parser.add_argument("--size", help="Image size (default=%(default)s).", type=float, default=None)
    parser.add_argument("--confidence-threshold", help="Confidence threshold (default=%(default)s).", type=float, default=None)
    parser.add_argument("--nms-threshold", help="NMS threshold (default=%(default)s).", type=float, default=None)
    parser.add_argument("--extensions", help="Target extensions split by commas (default=%(default)s).", default="jpg,png,jpeg")
    
    args = parser.parse_args()
    
    # Disable MXNET CUDNN autotuning since the images come in different sizes.
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    with open(args.detector_path, "r") as f:
        config = json.load(f)
    if args.use_gpu > 0:
        use_gpu = True
    else:
        use_gpu = False
    detector = LFFDDetector(config, use_gpu=use_gpu)
    
    extensions = args.extensions.split(",")
    paths = list(more_itertools.flatten([glob.glob(os.path.join(args.input_directory, "**", f"*.{ext}"), recursive=True) for ext in extensions]))
    print(f"{time.ctime()}: Start inferencing. Press `q` to cancel. Press  `-` to go back.")
    idx = 0
    while True:
        path = paths[idx]
        image = cv2.imread(path)
        start = time.time()
        boxes = detector.detect(image, size=args.size, confidence_threshold=args.confidence_threshold, nms_threshold=args.nms_threshold)
        elapsed = time.time() - start
        drawn_image = detector.draw(image, boxes)
        drawn_image = imutils.resize(drawn_image, height=args.display_height)
        cv2.imshow("Image", drawn_image)
        print(f"{time.ctime()}: [{idx:>7}] Inference time: {elapsed:.4f} seconds. Path: {path}")
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("-"):
            idx = max(idx - 1, 0)
        else:
            idx += 1
            if idx >= len(paths):
                break
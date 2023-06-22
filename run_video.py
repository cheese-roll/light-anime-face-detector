"""
    Run detector on the video and save the output as another video.
"""

import cv2
import json
import time
import argparse

from tqdm import tqdm
from collections import defaultdict

from core.detector import LFFDDetector

def save_output(video_path, output_path, detector, size=None, confidence_threshold=None, nms_threshold=None, roi=(), frame_count=None, run_per_x_frames=1, fps=None):
    """
        Run detector on the video and save the output as another video.
        Args:
            video_path: Path to the source video.
            output_path: Path to the output video.
            detector: A LFFDDetector instance.
            size: Image size (longer side) for the detector.
            confidence_threshold: Minimum confidence threshold for the detector.
            nms_threshold: NMS threshold for the detector.
            roi: A region of interest in a format of tuple of (xmin, ymin, xmax, ymax).
            frame_count: Number of frames to detect (including skipped frames).
            run_per_x_frames: Run an inference every X frames.
            fps: Output video FPS.
        Returns:
            A performance dict.
    """
    cap = cv2.VideoCapture(video_path)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) / run_per_x_frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if roi:
        xmin, ymin, xmax, ymax = roi
        frame_width, frame_height = xmax - xmin, ymax - ymin
    else:
        xmin = 0
        ymin = 0
        xmax = frame_width
        ymax = frame_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    fc = frame_count or fc
    frame_count = fc
    bar = tqdm(total=int(frame_count))
    timer = defaultdict(float)
    print(f"{time.ctime()}: Start.")
    total_start = time.time()
    while True:
        start = time.time()
        ret = cap.grab()
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret or frame_idx > frame_count:
            break
        if int(frame_idx) % run_per_x_frames != 0:
            bar.update()
            continue
        _, frame = cap.retrieve()
        timer["decode"] += time.time() - start
        
        start = time.time()
        frame = frame[ymin:ymax, xmin:xmax]
        timer["crop"] += time.time() - start
        
        start = time.time()
        boxes = detector.detect(frame, size=size, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold)
        timer["detect"] += time.time() - start
        
        start = time.time()
        frame = detector.draw(frame, boxes, thickness=3, font_scale=1)
        timer["draw"] += time.time() - start
        
        start = time.time()
        out.write(frame)
        timer["write"] += time.time() - start
        
        timer["count"] += 1
        bar.update()
    bar.close()
    cap.release()
    out.release()
    timer["total (incl. all)"] = time.time() - total_start
    print(f"{time.ctime()}: Finished. ({time.time()-total_start:.4f} seconds.)")
    return timer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run detector on the video.")
    parser.add_argument("-i", "--input-path", help="Input video path.", required=True)
    parser.add_argument("-o", "--output-path", help="Output video path.", required=True)
    parser.add_argument("-d", "--detector-path", help="Detector JSON configuration path.", required=True)
    parser.add_argument("--use-gpu", help="Use GPU or not (default=%(default)s).", type=int, default=1)
    parser.add_argument("--size", help="Image size (default=%(default)s).", type=float, default=None)
    parser.add_argument("--confidence-threshold", help="Confidence threshold (default=%(default)s).", type=float, default=None)
    parser.add_argument("--nms-threshold", help="NMS threshold (default=%(default)s).", type=float, default=None)
    parser.add_argument("--fps", help="Video output FPS (default=%(default)s).", type=float, default=None)
    parser.add_argument("--run-per-x-frames", help="Run per X frames (default=%(default)s).", type=float, default=1)
    parser.add_argument("--frame-count", help="Number of video frames to process (including skipped ones) (default=%(default)s).", type=float, default=None)
    
    args = parser.parse_args()

    with open(args.detector_path, "r") as f:
        config = json.load(f)
    if args.use_gpu > 0:
        use_gpu = True
    else:
        use_gpu = False
    detector = LFFDDetector(config, use_gpu=use_gpu)
    perf = save_output(
        args.input_path, 
        args.output_path, 
        detector, 
        size=args.size,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        frame_count=args.frame_count, 
        run_per_x_frames=args.run_per_x_frames, 
        fps=args.fps
    )
    print("\n" + "=" * 25, "Performance", "=" * 25 + "\n")
    count = perf["count"]
    for k, v in perf.items():
        if k == "count":
            continue
        print(f"{k.upper():<30}: {v:.4f} seconds ({count / v:.4f} FPS)")
    print("\nNOTE: If the times do not add up to `TOTAL (INCL. ALL)`, it it because the total includes time wasted on grabbing frames that were skipped.")
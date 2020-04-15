# Light (and Fast) Anime Face Detector

![Demo GIF](/assets/demo.gif)

This repository is based on [a light and fast face detector (LFFD)](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices) ([paper](https://arxiv.org/abs/1904.10633)). The model is trained on data from [Anime-Face-Detector](https://github.com/qhgz2013/anime-face-detector) repository by qhgz2013 plus additional ~600 pseudo semi-supervised images sourced from [danbooru2019](https://www.gwern.net/Danbooru2019). This detector can theoretically run at >100 FPS while retaining an acceptable level of accuracy.

This repository is tested under the following settings:
- Ubuntu 18.04.4 LTS
- Python 3.6.8
- GeForce GTX 1060 6GB (Mobile)
- NVIDIA Driver 440.64
- CUDA 10.0.130 (installed via conda as cudatoolkit) 
- CUDNN 7.6.5 (installed via conda)
- mxnet 1.6.0

## Core Dependencies

- Python 3.6+
- opencv-contrib-python
- mxnet
- numpy

You might need other libraries (e.g. tqdm, imutils, etc.) in order to run the demo. If you want to utilize your GPU, please consult [MXNet's official installation guide](https://mxnet.apache.org/get_started).

## Usage

Run demo on a video from command line:
```
    python run_video.py --input-path INPUT_PATH --output-path OUTPUT_PATH --detector-path configs/anime.json
```

Run demo on images in a directory from command line:
```
    python run_directory.py --input-directory INPUT_DIR --detector-path configs/anime.json
```

As a Python library:

```
    import os
    from core.detector import LFFDDetector
    
    """
        !!! IMPORTANT !!!
        Disable auto-tuning
        You might experience a major slow-down if you run the model on images with varied resolution / aspect ratio.
        This is because MXNet is attempting to find the best convolution algorithm for each input size, 
        so we should disable this behavior if it is not desirable.
    """ 
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    detector = LFFDDetector(config, use_gpu=True)
    image = cv2.imread(IMAGE_PATH)
    boxes = detector.detect(image)
```

## Examples

![Demo Image](/assets/touhou-cannonball.jpg)
<p align="center">Credit: <a href="https://touhoucannonball.com/">Promotional art from Touhou Cannonball</a></p>

![Demo Image](/assets/demo2.jpg)
<p align="center">(Size 640) Credit: <a href="https://www.pixiv.net/en/artworks/77538304">素敵な墓場で暮しましょ！</a> by <a href="https://www.pixiv.net/en/users/132450">syuri22@例大祭た14a</a></p>

![Demo Image](/assets/demo.jpg)
<p align="center">Credit: <a href="https://www.pixiv.net/en/artworks/76553042">Me</a></p>

## Inference Time

The inference speed for this detector is varied depending on the `size` (or `resize_scale`) parameter in the config. By lowering the parameters, there can be a significant speed gain with potentially worse model performance. 

`Input Resolution` in the following table is after the image resizing is done. `Inference Time` is calculated by timing `LFFDDetector.detect()` method.

Base | Input Resolution (WxH) | Inference Time
--- | --- | --- 
Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 (CPU) | 384x259 | 115ms
Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 (CPU) | 384x216 | 96ms
Intel® Core™ i7-8750H CPU @ 2.20GHz × 12 (CPU) | 276x384 | 125ms
GeForce GTX 1060 6GB (Mobile) | 384x259 | 13.7ms
GeForce GTX 1060 6GB (Mobile) | 384x216 | 10.6ms
GeForce GTX 1060 6GB (Mobile) | 276x384 | 17.7ms

MXNet auto-tuning is disabled for GPU benchmarking (by setting `MXNET_CUDNN_AUTOTUNE_DEFAULT` to `0` in environment variable). There is a potential gain by enabling it in case that the detector runs on images with the same resolution / aspect ratio.

## Training 

Please refer to [the original LFFD implementation](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices) for details. 

## Known Issues / Improvements

Currently, the detector cannot handle these cases well.

- Extreme size (the whole image is a face or extremely small face)
- "Non-standard" (e.g. chibi) stylistic choices
- Rotation
- Side views
- Ornaments / headdresses (e.g. helmet)
- Faces that are very close to others

All of these except the last one could be solvable by using more data and / or image augmentation. We could let the detector pre-label the data for us, which should reduce the workload when we manually correct them afterwards.

A consistent rule for bounding box might also be needed to minimize the bounding box loss (right now, it's a rough estimation of how much padding we will need for a face).

`size` (and `resize_scale`) also needs to be carefully chosen since the trained dataset is not big enough for the model to learn continuous face scaling.

Speed-wise, we could implement a batch processing. Non-maximum suppression function could also be written as `cython` or to be compatiable with `numba` to gain more speed.
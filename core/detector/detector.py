import cv2
import json
import mxnet as mx
from core.detector import LFFDPredictor

class LFFDDetector(object):
    def __init__(self, config, use_gpu=False):
        self.config = config
        with open(self.config["lffd_config_path"], "r") as f:
            lffd_config = json.load(f)
        self.predictor = LFFDPredictor(
            mxnet=mx,
            symbol_file_path=self.config["symbol_path"],
            model_file_path=self.config["model_path"],
            ctx=mx.cpu() if not use_gpu else mx.gpu(0),
            **lffd_config
        )
        
    @classmethod
    def draw(self, image, boxes, color=(0, 255, 0), font_scale=0.3, thickness=1):
        """
            Draw boxes on the image. This function does not modify the image in-place.
            Args:
                image: A numpy BGR image.
                boxes: A list of dict in the same format as returned from `LFFDDetector.detect` function.
                colors: Color (BGR) used to draw.
                font_scale: Font size for the confidence level display.
                thickness: Thickness of the line.
            Returns:
                A drawn image.
        """
        image = image.copy()
        for box in boxes:
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            confidence = box["confidence"]
            label = f"{confidence * 100:.2f}%"
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
        return image
    
    @classmethod
    def _parse(self, box):
        """
            Parse a predictor tuple to dict.
            Args:
                box: A tuple in order of (xmin, ymin, xmax, ymax, confidence)
            Returns:
                A dict with `xmin`, `ymin`, `xmax`, `ymax`, and `confidence` keys.
        """
        return {
            "xmin": int(box[0]),
            "ymin": int(box[1]),
            "xmax": int(box[2]),
            "ymax": int(box[3]),
            "confidence": box[4]
        }
        
    def detect(self, image, size=None, resize_scale=None, confidence_threshold=None, nms_threshold=None):
        """
            Detect objects from the given BGR image (numpy array)/
            Args:
                image: A numpy BGR image.
                size: 
                    Target size (longer side) for the image to rescale to before being fed to the net. 
                    This value is not necessary equal to the input size of the model.
                    If this value is None, it will be derived from the config.
                resize_scale: Resizing scale for the image. If this value is None, it will be derived from the `size` parameter instead.
                confidence_threshold: Minimum confidence threshold. if this value is None, it will be derived from the config.
                nms_threshold: NMS IOU threshold. If this value is None, it will be derived from the config.
            Returns:
                A list of dict of detections with `xmin`, `ymin`, `xmax`, `ymax`, and `confidence` keys.
        """
        if resize_scale is None:
            size = size or self.config["size"]
            h, w, _ = image.shape
            resize_scale = min((size / max(h, w)), 1)
        confidence_threshold = confidence_threshold or self.config["confidence_threshold"]
        nms_threshold = nms_threshold or self.config["nms_threshold"]
        bboxes, _ = self.predictor.predict(
            image, 
            resize_scale=resize_scale, 
            score_threshold=confidence_threshold, 
            top_k=10000, 
            NMS_threshold=nms_threshold, 
            NMS_flag=True, 
            skip_scale_branch_list=[]
        )
        return [self._parse(box) for box in bboxes]
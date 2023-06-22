# coding: utf-8
"""
    Modified version of the predictor from LFFD repository.
"""

import sys
import os
import numpy
import cv2
import time

class DataBatch:
    pass

def NMS(boxes, overlap_threshold):
    '''
    :param boxes: numpy nx5, n is the number of boxes, 0:4->x1, y1, x2, y2, 4->score
    :param overlap_threshold:
    :return:
    '''
    if boxes.shape[0] == 0:
        return boxes

    if boxes.dtype != numpy.float32:
        boxes = boxes.astype(numpy.float32)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]
    widths = x2 - x1
    heights = y2 - y1

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = heights * widths

    idxs = numpy.argsort(sc)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # compare second highest score boxes
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bo（ box
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > overlap_threshold)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]

class LFFDPredictor(object):

    def __init__(self,
                 mxnet,
                 symbol_file_path,
                 model_file_path,
                 ctx,
                 receptive_field_list,
                 receptive_field_stride,
                 bbox_small_list,
                 bbox_large_list,
                 receptive_field_center_start,
                 num_output_scales
                 ):
        self.mxnet = mxnet
        self.symbol_file_path = symbol_file_path
        self.model_file_path = model_file_path
        self.ctx = ctx

        self.receptive_field_list = receptive_field_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.num_output_scales = num_output_scales
        self.constant = [i / 2.0 for i in self.receptive_field_list]
        self.input_height = 480
        self.input_width = 640
        self.__load_model()

    def __load_model(self):
        print(f"{time.ctime()}: Initializing.")
        start = time.time()
        self.symbol_net = self.mxnet.symbol.load(self.symbol_file_path)
        data_name = 'data'
        data_name_shape = (data_name, (1, 3, self.input_height, self.input_width))
        self.module = self.mxnet.module.Module(symbol=self.symbol_net,
                                               data_names=[data_name],
                                               label_names=None,
                                               context=self.ctx,
                                               work_load_list=None)
        self.module.bind(data_shapes=[data_name_shape],
                         for_training=False)

        save_dict = self.mxnet.nd.load(self.model_file_path)
        self.arg_name_arrays = dict()
        self.arg_name_arrays['data'] = self.mxnet.nd.zeros((1, 3, self.input_height, self.input_width), self.ctx)
        self.aux_name_arrays = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                self.arg_name_arrays.update({name: v.as_in_context(self.ctx)})
            if tp == 'aux':
                self.aux_name_arrays.update({name: v.as_in_context(self.ctx)})
        self.module.init_params(arg_params=self.arg_name_arrays,
                                aux_params=self.aux_name_arrays,
                                allow_missing=True)
        print(f"{time.ctime()}: Initialized ({time.time() - start:.4f} seconds).")

    def predict(self, image, resize_scale=1, score_threshold=0.8, top_k=100, NMS_threshold=0.3, NMS_flag=True, skip_scale_branch_list=[]):
        if image.ndim != 3 or image.shape[2] != 3:
            return None

        bbox_collection = []

        shorter_side = min(image.shape[:2])
        if shorter_side * resize_scale < 128:
            resize_scale = float(128) / shorter_side
        
        if resize_scale != 1:
            input_image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        else:
            input_image = image

        input_image = input_image.astype(dtype=numpy.float32)
        input_image = input_image[:, :, :, numpy.newaxis]
        input_image = input_image.transpose([3, 2, 0, 1])

        data_batch = DataBatch()
        data_batch.data = [self.mxnet.ndarray.array(input_image, self.ctx)]
        
        tic = time.time()
        self.module.forward(data_batch=data_batch, is_train=False)
        results = self.module.get_outputs()
        outputs = []
        for output in results:
            outputs.append(output.asnumpy())
        toc = time.time()
        infer_time = (toc - tic) * 1000

        for i in range(self.num_output_scales):
            if i in skip_scale_branch_list:
                continue

            score_map = numpy.squeeze(outputs[i * 2], (0, 1))
            bbox_map = numpy.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
            RF_center_Xs_mat = numpy.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = numpy.array([self.receptive_field_center_start[i] + self.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
            RF_center_Ys_mat = numpy.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.constant[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.constant[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.constant[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.constant[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image.shape[1]] = image.shape[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image.shape[0]] = image.shape[0]

            select_index = numpy.where(score_map > score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > top_k:
            bbox_collection = bbox_collection[0:top_k]
        bbox_collection_numpy = numpy.array(bbox_collection, dtype=numpy.float32)

        if NMS_flag:
            final_bboxes = NMS(bbox_collection_numpy, NMS_threshold)
            final_bboxes_ = []
            for i in range(final_bboxes.shape[0]):
                final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3], final_bboxes[i, 4]))
            return final_bboxes_, infer_time
        else:
            return bbox_collection_numpy, infer_time
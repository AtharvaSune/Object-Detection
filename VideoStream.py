import os
import pathlib
import sys
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as util_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


util_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url+model_file,
                                        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_categories_from_labelmap(PATH_TO_LABELS,
                                                                use_display_name=True)
category_idx = {i+1: val for i, val in enumerate(category_index)}
print(category_index)
COLORS = np.random.randint(0, 255, size=(len(category_index), 3), dtype='uint8')
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


def detect(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = util_ops.reframe_box_masks_to_image_masks(
                                        output_dict['detection_masks'],
                                        output_dict['detection_boxes'],
                                        image.shape[0],
                                        image.shape[1]
                                    )
        detection_masks_reframed = tf.cast(
                                        detection_masks_reframed > 0.5,
                                        tf.uint8
                                    )
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, cap):
    while True:
        ret, image_np = cap.read()
        (H, W) = image_np.shape[:2]
        output_dict = detect(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_idx,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(25) & 0xFF == ord('c'):
            cv2.imwrite("Capture1.jpeg", image_np)


def run_model(model):
    cap = cv2.VideoCapture(0)
    show_inference(model, cap)


run_model(detection_model)
print("[DONE].............................\n")

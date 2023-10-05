import os
import cv2


def Test():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_caffemodel_path = os.path.join(
        script_dir, '__generics__', 'generic_data.caffemodel')
    model_prototxt_path = os.path.join(
        script_dir, '__generics__', 'generic_data.prototxt')
    model = cv2.dnn.readNet(model_caffemodel_path, model_prototxt_path)

    return model

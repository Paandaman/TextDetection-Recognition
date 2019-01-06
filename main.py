import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import visualization_utils

class Text_Detector(object):

    def __init__(self):
        PATH_TO_MODEL = 'path/to/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num

if __name__ == "__main__":

    detector = Text_Detector()

    img = "path/to/img.png"
    img_tmp = Image.open(img)
    image_to_array =np.asarray(img_tmp)
    img = image_to_array[:, :, 0:3] # Remove image transparancy channel
    boxes, scores, classes, nums = detector.get_classification(img)
    box = boxes[0][0] # Get box with most certainty

    visualization_utils.draw_bounding_box_on_image(img_tmp, box[0], box[1], box[2], box[3])
    img_tmp.show()



























































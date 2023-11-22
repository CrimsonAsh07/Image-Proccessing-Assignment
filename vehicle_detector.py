import cv2
import numpy as np
import gdown
import os 

class VehicleDetector:

    def __init__(self):
        # Define the path to the weights file
        output_path = "dnn_model/yolov4.weights"

        # Check if weights file already exists
        if not os.path.exists(output_path):
            weights_url = "https://drive.google.com/uc?id=17pMmbjt9WuWjrCAGFm_eFaCKsZ0J0kt4"
            gdown.download(weights_url, output_path, quiet=False)


        # Load Network
        net = cv2.dnn.readNet(output_path, "dnn_model/yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(832, 832), scale=1 / 255)


        # Allow classes containing Vehicles only
        self.classes_allowed = [2, 3, 5, 6, 7]


    def detect_vehicles(self, img):
        # Detect Objects
        vehicles_boxes = []
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        for class_id, score, box in zip(class_ids, scores, boxes):
            if score < 0.5:
                # Skip detection with low confidence
                continue

            if class_id in self.classes_allowed:
                vehicles_boxes.append(box)

        return vehicles_boxes


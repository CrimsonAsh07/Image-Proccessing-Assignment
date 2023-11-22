import cv2
from vehicle_detector import VehicleDetector
# Load Veichle Detector
vd = VehicleDetector()
img = cv2.imread('./images/pexels-charles-kettor-1077785.jpg')
vehicle_boxes = vd.detect_vehicles(img)
for box in vehicle_boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
cv2.imshow("Cars", img)
cv2.waitKey(0)
vehicle_count = len(vehicle_boxes)
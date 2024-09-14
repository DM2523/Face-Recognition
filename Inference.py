#imports
import yaml
from yaml.loader import SafeLoader
import cv2
import os
import numpy as np


class YOLO_PRED():

    def __init__(self,onn_model,data_yaml):

        #load config file
        with open(data_yaml,mode='r') as f:
            dataset_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = dataset_yaml['names']
        self.nc = dataset_yaml['nc']


        #loading inferencing model
        self.model = cv2.dnn.readNetFromONNX(onn_model)

        #set backends
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Function to pad the image to a square
    def pad_to_square(self,image):
        h, w = image.shape[:2]

        # Determine the size of the square (max of height   and width)
        new_size = max(h, w)

        # Compute padding
        top = (new_size - h) // 2
        bottom = new_size - h - top
        left = (new_size - w) // 2
        right = new_size - w - left

        # Pad the image
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        return padded_image, top, left

    def predict(self,image):
        # Pad the input image
        input_image, top_padding, left_padding = self.pad_to_square(image)

        # Conversion to blob
        INPUT_WH = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255,(INPUT_WH, INPUT_WH), swapRB=True, crop=False)

        # Inference
        self.model.setInput(blob)
        output = self.model.forward()
        output = output.transpose((0, 2, 1))

        # Extract output detection
        class_ids, confs, boxes = list(), list(), list()

        padded_height, padded_width, _ = input_image.shape

        # Scaling factors based on the padded image
        x_factor = padded_width / INPUT_WH
        y_factor = padded_height / INPUT_WH

        rows = output[0].shape[0]  # Number of rows,
        for i in range(rows):
            row = output[0][i]

            class_scores = row[4:]

            # Find the class with the highest score
            _, _, _, max_idx = cv2.minMaxLoc(class_scores)
            class_id = max_idx[1]

            # Get the confidence of the class (max class        score)
            class_conf = class_scores[class_id]

            # Only consider the detection if the class      confidence is high enough
            if class_conf > 0.25:  # Set confidence threshold
                confs.append(float(class_conf))
                label = int(class_id)
                class_ids.append(label)

                # Extract bounding box coordinates from model       output
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor) -      left_padding
                top = int((y - 0.5 * h) * y_factor) -       top_padding
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = list(), list(), list()

        # Apply Non-Max Suppression (NMS) to remove duplicate       boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.30)

        person_boxes = []

        # Check if indexes are not empty
        if len(indexes) > 0:
            for i in indexes.flatten():  # Flatten the      indexes to access values correctly
                r_class_ids.append(class_ids[i])
                r_confs.append(confs[i])
                r_boxes.append(boxes[i])

                # Draw bounding boxes
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                

                # Define the text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text = f'{self.labels[class_ids[i]]}: {round(confs[i]*100,0)}%'
                colors = self.generate_colors(class_ids[i])

                # Draw the rectangle
                cv2.rectangle(image, (left, top), (left + width, top + height), colors,    thickness=5, lineType=cv2.LINE_AA)

                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # Draw a filled rectangle behind the text 
                cv2.rectangle(image, (left, top - text_height - 10), (left + text_width, top), colors, cv2.FILLED)
                if self.labels[class_ids[i]]=='person':
                    person_boxes.append([(left, top), (left + width, top + height)])

                # Add the text on top of the bounding box
                cv2.putText(image, text, (left, top - 5), font, font_scale, (0, 0, 0), font_thickness)

        return image,person_boxes
    
    def generate_colors(self,ID):
        np.random.seed(42)
        colors = np.random.randint(100,255,size=(self.nc,3),dtype='uint8').tolist()
        return tuple(colors[ID])

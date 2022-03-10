import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

COLORS = [[200,0,0],[0,0,200]]

#Inputs
#boxes: torch.tensor, dtype=torch.float
#boxespred: torch.tensor, dtype=torch.float
##Image: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x853 at 0x1F5CDF7BB80>

def draw_boxes(boxes, boxespred, image, save_name):
    # read the image with OpenCV
    image = cv2.cvtColor(image*255, cv2.IMREAD_COLOR)
    for i, box in enumerate(boxes):
        color = COLORS[0]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, "truth", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, 
                    lineType=cv2.LINE_AA)
    for i, box in enumerate(boxespred):
        color = COLORS[1]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, "prediction", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 
                    lineType=cv2.LINE_AA)
    print('saving image')
    cv2.imwrite(f"{save_name}.jpg", image)
    print('SAVED!')
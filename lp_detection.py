import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import os
import easyocr


def processimage(img):
    # Resize its shape to fit the model input requirements.
    img = cv2.resize(img, (300, 300))

    #img = img.transpose(2, 0, 1)
    input_img = np.expand_dims(img, axis=0)
    return input_img

def text_extraction(im, lang_code='en'):
    reader = easyocr.Reader([lang_code], gpu=False)
    output = reader.readtext(im)
    return output

def lpdetect1(input_img):
    # A directory where the model will be downloaded.
    base_model_dir = "lpd_model"
    precision = "FP16"
    # The name of the model from Open Model Zoo.
    model_name =  "vehicle-license-plate-detection-barrier-0106"

    # The output path for the conversion
    model_path = f"{base_model_dir}/intel/{model_name}/{precision}/{model_name}.xml"

    # Initialize OpenVINO Runtime
    ie_core = Core()
    # Read the network and corresponding weights from a file.
    model = ie_core.read_model(model=model_path)
    # Compile the model for the CPU (you can choose manually CPU, GPU, MYRIAD etc.)
    # or let the engine choose the best available device (AUTO).
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")

    # Get input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Get the input size.
    input_height, input_width = list(input_layer.shape)[2:4]

    p_img = processimage(input_img )


    # Get the result.
    # result = compiled_model([input_img, auxiliary_blob])[output_layer]
    result = compiled_model([p_img])[output_layer]
    result = result.reshape(-1,7)
    flag = 0
    imgs = input_img.copy()
    for detection in result:
        confidence = float(detection[2])
        xmin = int(detection[3] * input_img.shape[1])
        ymin = int(detection[4] * input_img.shape[0])
        xmax = int(detection[5] * input_img.shape[1])
        ymax = int(detection[6] * input_img.shape[0])
        eps = 10
        if confidence > 0.5 and (ymax-ymin)<100:
            flag =1
            cim = input_img[ymin:ymax+eps,xmin:xmax+eps]
            cv2.imwrite("cim.jpg",cim)
            input_img = imgs.copy()
            cv2.rectangle(input_img, (xmin, ymin), (xmax+eps, ymax+eps), color=(0, 0, 255), thickness=2)
            

    return input_img,flag





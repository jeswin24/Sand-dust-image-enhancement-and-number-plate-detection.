from flask import Flask, render_template,request
import cv2
import numpy as np
import os
import time
from enhance_im import enhance
from lp_detection import lpdetect1,text_extraction
from haarcascade_lp_detection import lpdetect2



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/detect", methods=['POST','GET'])
def predict():
    if request.method=='POST':
        file = request.files['file']
        if os.path.exists("static\cim.jpg"):
            os.remove("static\cim.jpg")
        if file:
            filename = file.filename
            path = os.path.join('static',filename)
            file.save(path)
            txt =""
            time.sleep(2)
            src = cv2.imread(path)
            # enhance image
            E = enhance(src)
            E = np.uint8(E*255)
            enhfilename = "enhanced.jpg"
            cv2.imwrite(os.path.join('static',enhfilename),E)
            # detect licence plate
            D,flag = lpdetect1(E)
            if not flag:
                D = lpdetect2(E)
            outfilename = "output.jpg"
            
            filepath = os.path.join('static',outfilename )
            cv2.imwrite(filepath,D)
            print("image saved")
            lp = text_extraction(D)
            
            txt = lp[0][1]
                
            
            return render_template('index.html', msg=txt,filename1=filename,filename2=enhfilename,filename3=outfilename)
        else:
            return render_template('index.html', msg="No file")

    elif request.method=='GET':
        return render_template('index.html', prediction_text="Get Method")



if __name__ == "__main__":
    app.run(debug=True)
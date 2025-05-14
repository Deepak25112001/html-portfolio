from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
import base64

app = Flask("C:/Users/tadil/OneDrive/Desktop/New folder (2)/beject detection finall")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB limit

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model files
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
with open('Labels.txt', 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')

# Configure the model
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return "Failed to load image", 400

        classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
        if isinstance(classIndex, tuple):  # No detection
            classIndex = []
            confidence = []
            bbox = []

        for classId, conf, box in zip(np.array(classIndex).flatten(),
                                      np.array(confidence).flatten(), bbox):
            label = classLabels[classId - 1] if classId <= len(classLabels) else "Unknown"
            cv2.rectangle(img, box, (255, 0, 0), 2)
            cv2.putText(img, label, (box[0]+10, box[1]+40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        output_path = os.path.join('static', 'output.jpg')
        cv2.imwrite(output_path, img)

        return redirect(url_for('result', output_file='output.jpg'))

    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/webcam_detect', methods=['POST'])
def webcam_detect():
    try:
        img_data = request.form['image']
        img_data = img_data.split(',')[1]
        img_array = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        classIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
        if isinstance(classIndex, tuple):
            classIndex = []
            confidence = []
            bbox = []

        for classId, conf, box in zip(np.array(classIndex).flatten(),
                                      np.array(confidence).flatten(), bbox):
            label = classLabels[classId - 1] if classId <= len(classLabels) else "Unknown"
            cv2.rectangle(img, box, (255, 0, 0), 2)
            cv2.putText(img, label, (box[0]+10, box[1]+40), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        output_path = os.path.join('static', 'output.jpg')
        cv2.imwrite(output_path, img)

        return redirect(url_for('result', output_file='output.jpg'))

    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('webcam'))

@app.route('/result')
def result():
    output_file = request.args.get('output_file')
    return render_template('result.html', output_file=output_file)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template, Response
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import base64

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8m_custom.pt') # or yolo8m.pt model can be used...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' in request.files:
        # Get the image from the request
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Run inference
        results = model(image)
        
        # Process detection results
        image_with_boxes, num_boxes, labels = draw_boxes(image, results)
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image_with_boxes)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response = {
            'image': encoded_image,
            'num_boxes': num_boxes,
            'labels': labels
        }
        return jsonify(response)

@app.route('/detect_video')
def detect_video():
    return render_template('detect_video.html')

@app.route('/video_feed')
def video_feed():
    video_url = request.args.get('video_url')
    return Response(stream_video(video_url), mimetype='multipart/x-mixed-replace; boundary=frame')

def draw_boxes(image, results):
    num_boxes = len(results[0].boxes)
    labels = [model.names[int(box.cls)] for box in results[0].boxes]
    # Draw boxes on the image
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy)
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     label = f'{model.names[int(box.cls)]} {box.conf:.2f}'
    #     cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for box in results[0].boxes:
        xyxy = box.xyxy.tolist()[0]  # Get the first (and only) row
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{model.names[int(box.cls)]} {box.conf.item():.2f}'  # Convert tensor to scalar using .item()
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Add label and number of boxes text
    text = f'Detected: {num_boxes}'
    y0, dy = 30, 30
    for i, line in enumerate(labels):
        y = y0 + i * dy
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, text, (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image, num_boxes, labels

def stream_video(video_url):
    cap = cv2.VideoCapture(video_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame)
        
        # Process detection results
        frame_with_boxes, num_boxes, labels = draw_boxes(frame, results)
        
        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame_with_boxes)
        frame = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)

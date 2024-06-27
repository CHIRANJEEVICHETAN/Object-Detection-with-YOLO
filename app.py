from flask import Flask, request, jsonify, render_template, Response
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import base64
from scipy.spatial import distance as dist
from imutils import perspective, contours
import imutils
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Global variable to store dimensions list from video feed
dimensions_list_video = []

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8m_custom.pt')

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def measure_objects(image, width, pixelsPerMetric):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    objects = []

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv3() or imutils.is_cv4() else cv2.cv.BoxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        objects.append((orig, dimA, dimB, c))

    return objects, pixelsPerMetric

@app.route('/')
def index():
    return render_template('index.html')

# Inside the detect_image function, add logging
@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' in request.files and 'width' in request.form:
        file = request.files['image']
        width = float(request.form['width'])  # Convert width to float
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(image)
        image_with_boxes, num_boxes, labels = draw_boxes(image, results)
        objects, _ = measure_objects(image_with_boxes, width, pixelsPerMetric=None)
        objects_data = [{'index': i, 'image': obj[0], 'dimA': obj[1], 'dimB': obj[2]} for i, obj in enumerate(objects)]
        
        # logging.debug(f"Objects Data: {objects_data}")
        # logging.debug(f"Labels: {labels}")
        
        if objects_data:
            encoded_images = [base64.b64encode(cv2.imencode('.jpg', obj['image'])[1]).decode('utf-8') for obj in objects_data]
            dimensions_list = get_dimensions_list(objects_data, labels)
            response = {
                'images': encoded_images,
                'num_boxes': num_boxes,
                'labels': labels,
                'dimensions_list': dimensions_list,
                'total_objects': len(objects_data)
            }
            return jsonify(response)
        return jsonify({'error': 'No objects detected'})

@app.route('/detect_video')
def detect_video():
    return render_template('detect_video.html')

@app.route('/video_feed')
def video_feed():
    video_url = request.args.get('video_url')
    width_str = request.args.get('width')

    try:
        width = float(width_str)  # Convert width to float
    except (TypeError, ValueError) as e:
        logging.error(f"Invalid width parameter: {width_str} - {e}")
        return jsonify({'error': 'Invalid width parameter'}), 400

    return Response(stream_video(video_url, width), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add an endpoint to retrieve dimensions list
@app.route('/dimensions_list')
def dimensions_list():
    global dimensions_list_video
    return jsonify(dimensions_list_video)


# Modify the get_dimensions_list function
def get_dimensions_list(objects_data, labels):
    dimensions_list = []
    for obj in objects_data:
        try:
            index = obj.get('index', None)
            if isinstance(index, int) and index < len(labels):
                label = labels[index]
            else:
                label = 'Unknown'
        except IndexError as e:
            logging.error(f"IndexError: {e}")
            label = 'Unknown'
        dimensions_list.append({'Object': label, 'Dimension_A': obj['dimA'], 'Dimension_B': obj['dimB']})
    return dimensions_list

def draw_boxes(image, results):
    num_boxes = len(results[0].boxes)
    labels = [model.names[int(box.cls)] for box in results[0].boxes]
    for i, box in enumerate(results[0].boxes):
        xyxy = box.xyxy.tolist()[0]
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{model.names[int(box.cls)]} {box.conf.item():.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        results[0].boxes[i].index = i
    text = f'Detected: {num_boxes}'
    y0, dy = 30, 30
    for i, line in enumerate(labels):
        y = y0 + i * dy
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, text, (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image, num_boxes, labels

def stream_video(video_url, width):
    global dimensions_list_video  # Declare global variable

    cap = cv2.VideoCapture(video_url)
    pixelsPerMetric = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_with_boxes, num_boxes, labels = draw_boxes(frame, results)
        objects, pixelsPerMetric = measure_objects(frame_with_boxes, width, pixelsPerMetric)
        objects_data = [{'index': i, 'contour': obj[3], 'dimA': obj[1], 'dimB': obj[2]} for i, obj in enumerate(objects)]
        
        # logging.debug(f"Objects Data: {objects_data}")
        # logging.debug(f"Labels: {labels}")

        try:
            dimensions_list = get_dimensions_list(objects_data, labels)
            dimensions_list_video = dimensions_list  # Update global dimensions list
            logging.debug(f"Dimensions List: {dimensions_list}")
        except Exception as e:
            logging.error(f"Error in get_dimensions_list: {e}")
            dimensions_list_video = []

        # Draw bounding boxes around detected objects (remove dimension text in video)
        for obj in objects_data:
            contour = obj['contour']
            cv2.drawContours(frame_with_boxes, [contour], -1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame_with_boxes)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()

if __name__ == '__main__':
    app.run(debug=True)

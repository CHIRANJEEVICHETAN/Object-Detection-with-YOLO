# Real-Time Object Detection with YOLOv8: Image and Video Processing Web Application

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Requirements](#requirements)
4. [Software and Tools](#software-and-tools)
5. [Dataset Preparation](#dataset-preparation)
6. [Configuration File](#configuration-file)
7. [File Structure](#file-structure)
8. [Model Setup](#model-setup)
9. [Training the Model](#training-the-model)
   - [Rename Trained Weights](#rename-trained-weights)
   - [Model Inference](#model-inference)
10. [Explanation of Code](#explanation-of-code)
11. [Usage](#usage)
12. [Output Explanation](#output-explanation)
13. [Applications](#applications)
14. [Improved Features](#improved-features)
15. [Future Scope](#future-scope)
16. [Contributing](#contributing)
17. [License](#license)
18. [Summary](#summary)
19. [Results](#results-snapshots)
20. [References](#references)

## Introduction

This project involves developing a web application for real-time object detection using YOLOv8. The application can process both images and videos, providing real-time detection and counting of objects. The project includes dataset preparation, model training, and deployment as a web application.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- pip

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the YOLOv8 model:**

   Follow the steps in the [Model Setup](#model-setup) section.

4. **Run the application:**

   ```bash
   python app.py
   ```

   Open your browser and go to `http://127.0.0.1:5000/`.

## Requirements

- **Python 3.x**
- **OpenCV 4.x**
- **Ultralytics**
- **Pytorch**
- **Flask**
- **LabelImg**

## Software and Tools

- **Python**: Programming language used for scripting.
- **OpenCV**: Library for computer vision tasks.
- **Ultralytics**: Provides the YOLO object detection framework.
- **Pytorch**: Deep learning framework for building and training neural networks.
- **Flask**: Web framework for Python.
- **LabelImg**: Tool for annotating images.

## Dataset Preparation

1. **Annotate Images**: Use `LabelImg` to annotate images and create labels.
2. **Split Dataset**: Split the dataset into training and validation sets with separate `images` and `labels` directories.

## Configuration File

Create a `data_custom.yaml` file with the following configuration:

```yaml
train: /path/to/your/project/Datasets/Objects/train
val: /path/to/your/project/Datasets/Objects/val
nc: 1
names: ["box"]
```

## File Structure

- `Datasets/`
  - `Objects/`
    - `train/`
      - `images/`
      - `labels/`
    - `val/`
      - `images/`
      - `labels/`
- `runs/`
  - `detect/`
    - `predict/`
    - `train/`
- `static/`
  - `uploads/`
  - `style.css`
  - `spinner.gif`
- `templates/`
  - `index.html`
  - `detect_video.html`
- `app.py`
- `data_custom.yaml`
- `requirements.txt`
- `yolov8m.pt`
- `Dockerfile`
- `Git_Cloning.md`
- `README.md`
- `LICENSE`
- `yolo8m_custom.pt`

## Model Setup

1. **Download YOLOv8 Weights**: Download the `yolov8.pt` file from the Ultralytics GitHub repository.
2. **Install PyTorch**: Install PyTorch using the following command:
   ```bash
   pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Install Ultralytics**: Install the Ultralytics package:
   ```bash
   pip install ultralytics
   ```
4. **Activate Virtual Environment**: Activate your virtual environment.
5. **Verify Installation**: Check if PyTorch and CUDA are available:
   ```bash
   python -c "import torch; print(torch.__version__)"
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Training the Model

Train the YOLOv8 model using the following command:

```bash
yolo task=detect mode=train epochs=30 data=data_custom.yaml model=yolov8m.pt imgsz=640 batch=3
```

This process might take some time. Upon completion, you will get a `best.pt` file located inside `runs/detect/train/weights`.

### Rename Trained Weights

Rename the `best.pt` file to `yolov8m_custom.pt` and move it to the root directory.

### Model Inference

To detect objects in an image using the trained model, use the following command:

```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=1.jpg
```

## Explanation of Code

The `app.py` file contains the main code for the Flask web application. It handles the routes for image and video processing, utilizes the YOLOv8 model for detection, and serves the results to the frontend.

### Python Code for Box Detection

Use the following Python code in `detect_boxes.py` to load the trained model and detect objects in an image:

```python
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m_custom.pt')

# Load the image
image = cv2.imread('box3.jpg')
results = model(image)

# Draw bounding boxes on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Usage

### Image Processing

1. Go to the home page of the web application.
2. Upload an image file.
3. Click on the "Detect" button.
4. The application will display the image with detected objects.

### Video Processing

1. Go to the video processing page.
2. Enter the video URL.
3. The application will process the video and display it with detected objects.

## Output Explanation

After running the inference code, the output will be an image with detected boxes highlighted by green rectangles. The number of detected boxes will be displayed both on the image and printed in the console.

## Applications

The techniques and processes described in this project have several practical applications, including:

- **Automated Quality Control**: Detecting defects or missing components in manufacturing.
- **Logistics and Inventory Management**: Identifying and counting items in warehouses.
- **Surveillance and Security**: Monitoring and detecting objects in security footage.
- **Retail**: Managing stock and detecting product placement on shelves.

## Improved Features

- **YOLOv8**: Implemented the YOLOv8 model for image recognition, enhancing accuracy and performance in object detection tasks.
- **Real-Time Object Detection**: Integrated live video object detection, enabling the system to detect objects in real-time.
- **Flask Integration**: Compiled the entire project into a Flask application for seamless frontend and backend integration.
- **Drag and Drop Features**: Added drag and drop functionality for image uploads, providing a user-friendly experience.
- **Responsiveness**: Ensured the web application is responsive for seamless viewing on mobile devices.
- **Live Video**: Enabled object detection in live video feeds from external mobile devices using webcam IP addresses.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first. You can submit any bugs or feature requests as an issue, or make a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Summary

This project implements a web application for real-time object detection using the YOLOv8 model. It supports both image and video processing, providing accurate detection and counting of objects. The application includes dataset annotation, model training, and deployment as a web service. It leverages Flask for backend integration and offers practical applications in automated quality control, logistics, security monitoring, and retail management. Future enhancements could include multi-class detection and improved deployment options.

## Results (Snapshots)

![image](https://user-images.githubusercontent.com/102181527/209464438-5b)

## References

1. **YOLOv8 Documentation**:

   - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
   - [YOLOv8 Documentation](https://docs.ultralytics.com/)

2. **PyTorch**:

   - [PyTorch Official Website](https://pytorch.org/)
   - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

3. **OpenCV**:

   - [OpenCV Official Website](https://opencv.org/)
   - [OpenCV Documentation](https://docs.opencv.org/4.x/)

4. **Flask**:

   - [Flask Official Website](https://flask.palletsprojects.com/)
   - [Flask Documentation](https://flask.palletsprojects.com/en/2.0.x/)

5. **LabelImg**:

   - [LabelImg GitHub Repository](https://github.com/tzutalin/labelImg)
   - [LabelImg Installation and Usage](https://github.com/tzutalin/labelImg#installation)

6. **Docker**:

   - [Docker Official Website](https://www.docker.com/)
   - [Docker Documentation](https://docs.docker.com/)

7. **Python**:

   - [Python Official Website](https://www.python.org/)
   - [Python Documentation](https://docs.python.org/3/)

8. **Choreo.dev**:
   - [Choreo.dev Official Website](https://choreo.dev/)
   - [Choreo.dev Documentation](https://wso2.com/choreo/docs/)

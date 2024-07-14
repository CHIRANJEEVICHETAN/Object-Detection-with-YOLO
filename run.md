# Running the Real-Time Object Detection Web Application

Follow these steps to set up and run the Real-Time Object Detection web application using YOLOv8.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- pip

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch**: Install PyTorch using the following command:

   ```bash
   pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Download the YOLOv8 model weights:**

   Place the `yolov8m_custom.pt` file in the root directory of the project.

## Running the Application

1. **Run the Flask application:**

   ```bash
   python app.py
   ```

2. **Open your browser and go to:**

   ```
   http://127.0.0.1:5000/
   ```

## Using the Application

### Image Processing

1. Go to the home page.
2. Upload an image file.
3. Enter the width of the object in the image (in real-world units).
4. Click on the "Detect" button.
5. The application will display the image with detected objects and their dimensions.

### Video Processing

1. Go to the video processing page.
2. Enter the video URL and the width of the object in the video (in real-world units).
3. The application will process the video and display it with detected objects and their dimensions.

### Viewing Dimensions List

1. Click on the "View Dimensions" button to see the list of object dimensions detected from the video feed.

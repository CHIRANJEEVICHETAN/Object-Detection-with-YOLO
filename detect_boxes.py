import cv2
import numpy as np

def detect_boxes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding or edge detection to highlight the boxes
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image and retrieve the hierarchy
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store valid contours (boxes)
    valid_contours = []

    # Define minimum and maximum threshold values for box size
    min_area = 1000  # Adjust this value based on the minimum area of your boxes
    max_area = 5000  # Adjust this value based on the maximum area of your boxes

    # Iterate through all detected contours
    for i, contour in enumerate(contours):
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        
        # Calculate the aspect ratio
        aspect_ratio = float(w) / h
        
        # Filter contours based on area, aspect ratio, and hierarchy
        if area > min_area and area < max_area and aspect_ratio > 0.5 and aspect_ratio < 2.0:
            # Check if contour has no parent (top-level contour)
            if hierarchy[0][i][3] == -1:
                valid_contours.append(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Count the number of detected valid contours (boxes)
    num_boxes = len(valid_contours)

    # Display the result
    cv2.putText(image, f'Number of Boxes: {num_boxes}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Boxes Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the number of boxes detected
    print(f'Number of boxes detected: {num_boxes}')

# Example usage
detect_boxes('img2.JPEG')

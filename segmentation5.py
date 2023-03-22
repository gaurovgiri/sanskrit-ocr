import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load the Devanagari text image
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image to binary
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Remove matra
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(thresh, kernel, iterations=1)

# Find contours of the image
contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding rectangles of the contours
rectangles = [cv2.boundingRect(cnt) for cnt in contours]

# Sort the rectangles from left to right and top to bottom
rectangles = sorted(rectangles, key=lambda x: (x[1], x[0]))

# Segment words and individual letters
for i, rectangle in enumerate(rectangles):
    x, y, w, h = rectangle
    roi = thresh[y:y+h, x:x+w]
    
    # Find peaks in the histogram of the ROI to segment the letters
    hist = np.sum(roi, axis=0)
    peaks, _ = find_peaks(hist, distance=10)
    
    # Add the x-coordinate of the ROI to the peaks to get the actual x-coordinates of the letters
    peaks += x
    
    # Draw rectangles over the individual letters
    for j in range(len(peaks) - 1):
        x1, y1 = peaks[j], y
        x2, y2 = peaks[j+1], y+h
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
    # Draw rectangles over the words
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

# Show the image with rectangles
cv2.imshow("Output",erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
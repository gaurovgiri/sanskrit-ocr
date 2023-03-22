import cv2
import numpy as np

# Read the image
img = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilation = cv2.dilate(thresh, kernel, iterations=1)

erosion = cv2.erode(dilation, kernel, iterations=1)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over each contour
for i, cnt in enumerate(contours):
    # Get the bounding box coordinates of the contour
    x, y, w, h = cv2.boundingRect(cnt)

    # Ignore if the contour is too small or too large
    if w < 5 or h < 5 or w > img.shape[1] // 2 or h > img.shape[0] // 2:
        continue

    # Draw the bounding box around the contour
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract the letter
    letter = thresh[y:y+h, x:x+w]

    # Save the letter as an image file
    cv2.imwrite('letters/letter_{}.png'.format(i), letter)

cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

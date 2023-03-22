import cv2
import numpy as np
from scipy.signal import find_peaks

# Read the image
img = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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

    # Find the horizontal projection profile of the letter
    hor_proj = np.sum(letter, axis=0)

    # Find the peaks in the horizontal projection profile
    peaks, _ = find_peaks(hor_proj, distance=5,height=10)

    # Loop over the peaks
    for j, peak in enumerate(peaks[:-1]):
        # Crop the sub-letter
        sub_letter = letter[:, peak:peaks[j+1]]

        # Find the vertical projection profile of the sub-letter
        ver_proj = np.sum(sub_letter, axis=1)

        # Find the peaks in the vertical projection profile
        sub_peaks, _ = find_peaks(ver_proj, distance=5,height=10)

        # Loop over the sub-peaks
        for k, sub_peak in enumerate(sub_peaks[:-1]):
            # Create rectangular boundary around sub-letter
            x_sub = x + peak
            y_sub = y + sub_peak
            w_sub = peaks[j+1] - peak
            h_sub = sub_peaks[k+1] - sub_peak
            cv2.rectangle(img, (x_sub, y_sub), (x_sub + w_sub, y_sub + h_sub), (0, 0, 255), 2)

            # Crop the sub-sub-letter
            sub_sub_letter = sub_letter[sub_peak:sub_peaks[k+1], :]

            # Save the sub-sub-letter as an image file
            cv2.imwrite('letters/letter_{}_{}_{}.png'.format(i, j, k), sub_sub_letter)

cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

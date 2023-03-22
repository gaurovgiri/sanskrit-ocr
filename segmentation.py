import cv2

# Read the image
image = cv2.imread('2.jpg')


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to the grayscale image
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply morphological operations to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(threshold, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Find contours in the eroded image
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over each contour
for contour in contours:
    # Get the bounding box coordinates of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding box around the contour
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the contour and get the ROI (region of interest)
    roi = threshold[y:y + h, x:x + w]

    # Apply thresholding to the ROI
    _, roi_threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded ROI
    roi_contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour in the ROI
    for roi_contour in roi_contours:
        # Get the bounding box coordinates of the contour
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(roi_contour)
        
        # Ignore if the ROI is less than 20 as it might be noise
        if roi_w < 20:
            continue
        
        # Draw the bounding box around the contour
        cv2.rectangle(image, (x + roi_x, y + roi_y), (x + roi_x + roi_w, y + roi_y + roi_h), (0, 0, 255), 2)

        # Crop the contour and get the ROI (region of interest)
        roi2 = roi[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Apply thresholding to the ROI
        _, roi2_threshold = cv2.threshold(roi2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours in the thresholded ROI
        roi2_contours, _ = cv2.findContours(roi2_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over each contour in the ROI
        i = 1 
        for roi2_contour in roi2_contours:
            # Get the bounding box coordinates of the contour
            roi2_x, roi2_y, roi2_w, roi2_h = cv2.boundingRect(roi2_contour)

            # Draw the bounding box around the contour
            cv2.rectangle(image, (x + roi_x + roi2_x, y + roi_y + roi2_y), (x + roi_x + roi2_x + roi2_w, y + roi_y + roi2_y + roi2_h), (255, 0, 0), 2)
            # Extract the letter
            letter = roi2[roi2_y:roi2_y + roi2_h, roi2_x:roi2_x + roi2_w]
            
            # cv2.imwrite('letter_{}_{}_{}.png'.format(x + roi_x + roi2_x, y + roi_y + roi2_y, i), letter)
        
            i += 1

cv2.imshow('Output', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

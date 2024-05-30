import os
import cv2
import numpy as np

# Open source image file
IMAGES_DIR = os.path.join(r'C:\Users\Yukesh\Downloads', 'snookervideo')
image_path = os.path.join(IMAGES_DIR, 'balls.JPG')
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Convert image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image to black and white
thresh, image_edges = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)

# Create canvas
canvas = np.zeros(image.shape, np.uint8)
canvas.fill(255)

# Create background mask
mask = np.zeros(image.shape, np.uint8)
mask.fill(255)

# Create new background
new_background = np.zeros(image.shape, np.uint8)
new_background.fill(255)

# Get all contours
contours_draw, hierarchy = cv2.findContours(image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Get most significant contours
contours_mask, hierarchy = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw all contours
for contour in range(len(contours_draw)):
    # Draw current contour
    cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)

# Most significant contours traversal
for contour in range(len(contours_mask)):
    # Create mask
    if contour != 1:
        cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))

    # Create background
    if contour != 1:
        cv2.fillConvexPoly(new_background, contours_mask[contour], (0, 255, 0))

# Display the image in a window
cv2.imshow('Original', image)
cv2.imshow('Contours', canvas)
cv2.imshow('Background mask', mask)
cv2.imshow('New background', new_background)
cv2.imshow('Output', cv2.bitwise_and(image, new_background))

# Write images
cv2.imwrite('contours.png', canvas)
cv2.imwrite('mask.png', mask)
cv2.imwrite('background.png', new_background)
cv2.imwrite('output.png', cv2.bitwise_and(image, new_background))

# Escape condition
cv2.waitKey(0)

# Clean up windows
cv2.destroyAllWindows()

import cv2
import numpy as np

def stitch_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Use FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        raise ValueError("Not enough matches found to compute homography")

    # Extract matching keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get original image dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Find the corners of image1 after transformation
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, homography)

    # Get new width and height for the stitched image
    [min_x, min_y] = np.int32(transformed_corners.min(axis=0).ravel())
    [max_x, max_y] = np.int32(transformed_corners.max(axis=0).ravel())

    new_width = max(max_x, w2) - min(min_x, 0)
    new_height = max(max_y, h2) - min(min_y, 0)

    # Adjust homography to shift the warped image so that no parts are lost
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    adjusted_homography = translation_matrix @ homography

    # Warp image1 using the adjusted homography
    warped_image1 = cv2.warpPerspective(image1, adjusted_homography, (new_width, new_height))

    # Create an empty canvas for stitching
    stitched_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Place image2 in the correct position
    x_offset = -min_x if min_x < 0 else 0
    y_offset = -min_y if min_y < 0 else 0
    stitched_image[y_offset:y_offset + h2, x_offset:x_offset + w2] = image2

    # Overlay warped image1, preserving non-black pixels
    mask = (warped_image1 > 0)
    stitched_image[mask] = warped_image1[mask]

    return stitched_image

# Load images
image1 = cv2.imread('IMG_20250304_164212.jpg')
image2 = cv2.imread('IMG_20250304_164218.jpg')
image3 = cv2.imread('IMG_20250304_164226.jpg')
image4 = cv2.imread('IMG_20250304_164234.jpg')

# Stitch first two images
stitched1 = stitch_images(image1, image2)
cv2.imwrite('stitched_panorama1.jpg', stitched1)

# Stitch next two images
stitched2 = stitch_images(image3, image4)
cv2.imwrite('stitched_panorama2.jpg', stitched2)

# Stitch final panorama
final_panorama = stitch_images(stitched1, stitched2)
cv2.imwrite('final_panorama.jpg', final_panorama)

# Show final output
cv2.imshow('Final Stitched Panorama', final_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

def stitch_images(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

    if len(good_matches) < 4:
        raise ValueError("Not enough matches found to compute homography")

    # Extract matching keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    # Get dimensions of both images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Compute the transformed corners of image1
    corners_image1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_image1, homography)

    # Find the new bounding box for the panorama
    transformed_corners = transformed_corners.reshape(-1, 2)
    all_corners = np.vstack((transformed_corners, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])))
    [x_min, y_min] = np.int32(all_corners.min(axis=0).flatten())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).flatten())

    # Compute translation homography
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp image1 to fit in the panorama
    warp_size = (x_max - x_min, y_max - y_min)
    warped_image1 = cv2.warpPerspective(image1, translation @ homography, warp_size)

    # Place image2 in the correct position
    panorama = warped_image1.copy()
    panorama[-y_min:h2 - y_min, -x_min:w2 - x_min] = image2

    return panorama

# Load images
image1 = cv2.imread('IMG_20250304_164212.jpg')
image2 = cv2.imread('IMG_20250304_164218.jpg')
image3 = cv2.imread('IMG_20250304_164226.jpg')
image4 = cv2.imread('IMG_20250304_164234.jpg')

# Stitch images sequentially
stitched1 = stitch_images(image1, image2)
stitched2 = stitch_images(stitched1, image3)
final_panorama = stitch_images(stitched2, image4)

# Save final result
cv2.imwrite('final_panorama.jpg', final_panorama)

# Display final panorama
cv2.imshow('Final Stitched Panorama', final_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()

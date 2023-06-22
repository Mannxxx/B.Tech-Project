import cv2
import numpy as np

def extract_blade_tip(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (adjust the parameters as needed)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the contour with the largest area as the blade contour
    blade_contour = max(contours, key=cv2.contourArea)

    # Find the minimum enclosing circle of the blade contour
    (x, y), radius = cv2.minEnclosingCircle(blade_contour)
    center = (int(x), int(y))

    # Calculate the tip coordinates
    tip_coordinates = (int(x), int(y - radius))

    # Extract the blade region
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [blade_contour], 0, (255), thickness=cv2.FILLED)
    blade_image = cv2.bitwise_and(image, image, mask=mask)

    return tip_coordinates, blade_image

# Example usage
image_path = "C:/Users/Mansi/Desktop/BTP-1/Image-rotor.png"
tip_coordinates, blade_image = extract_blade_tip(image_path)
print("Blade tip coordinates:", tip_coordinates)

# Example usage
image_path = "C:/Users/Mansi/Desktop/BTP-1/rahul.png"
tip_coordinates, blade_image = extract_blade_tip(image_path)
print("Blade tip coordinates:", tip_coordinates)

# Display the extracted blade image
cv2.imshow("Extracted Blade Image", blade_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import numpy as np

# def extract_blade_tip(image_path):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply edge detection (adjust the parameters as needed)
#     edges = cv2.Canny(gray, threshold1=100, threshold2=200)

#     # Find contours in the edge image
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Select the contour with the largest area as the blade contour
#     blade_contour = max(contours, key=cv2.contourArea)

#     # Find the minimum enclosing circle of the blade contour
#     (x, y), radius = cv2.minEnclosingCircle(blade_contour)
#     center = (int(x), int(y))

#     # Calculate the tip coordinates
#     tip_coordinates = (int(x), int(y - radius))

#     return tip_coordinates


# # Example usage 1
# image_path = "C:/Users/Mansi/Desktop/BTP-1/Image-rotor.png"
# blade_tip = extract_blade_tip(image_path)
# print("Blade tip coordinates:", blade_tip)

# # Example usage 2
# image_path = "C:/Users/Mansi/Desktop/BTP-1/heli blade.png"
# blade_tip = extract_blade_tip(image_path)
# print("Blade tip coordinates:", blade_tip)

# # Example usage 3
# image_path = "C:/Users/Mansi/Desktop/BTP-1/apple.png"
# blade_tip = extract_blade_tip(image_path)
# print("Blade tip coordinates:", blade_tip)


# # Display the extracted blade image
# cv2.imshow("Extracted Blade Image", blade_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # def test_extract_blade_tip(image_path, expected_coordinates):
# #     # Extract the blade tip coordinates
# #     blade_tip = extract_blade_tip(image_path)

# #     # Compare the extracted coordinates with the expected coordinates
# #     if blade_tip == expected_coordinates:
# #         print(f"PASS: Blade tip coordinates for {image_path} are correct.")
# #     else:
# #         print(f"FAIL: Blade tip coordinates for {image_path} are incorrect. Expected: {expected_coordinates}, Got: {blade_tip}")

# # # Example usage
# # image_path_1 = "path/to/known_image_1.jpg"
# # expected_coordinates_1 = (100, 150)
# # test_extract_blade_tip(image_path_1, expected_coordinates_1)

# # image_path_2 = "path/to/known_image_2.jpg"
# # expected_coordinates_2 = (200, 250)
# # test_extract_blade_tip(image_path_2, expected_coordinates_2)

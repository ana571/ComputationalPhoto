import cv2
import numpy as np
import math
from PIL import Image
from IPython.display import display
from skimage.color import rgb2gray

def normalize_coordinate(x, y, width_equirect, height_equirect):
    x_prime = 2 * x / width_equirect - 1
    y_prime = 1 - 2 * y / height_equirect
    return (x_prime, y_prime)

def long_lat(x_prime, y_prime, lambda_0):
    longitude = np.pi * x_prime + lambda_0
    latitude = np.pi/2 * y_prime
    return (longitude, latitude)


def normalized_2D_fisheye_coord(latitude, longitude, aperture_in_rad, width, height, x, y):
    P_x = np.cos(latitude) * np.cos(longitude)
    P_y = np.cos(latitude) * np.sin(longitude)
    P_z = np.sin(latitude)

    theta = np.arctan2(P_z, P_x)

    sqrt_for_r = np.sqrt(P_x ** 2 + P_z ** 2)
    if (P_y != 0):
        r = 2 * np.arctan2(sqrt_for_r, P_y) / aperture_in_rad
    else:
        r = 0

    x_2 = (width - 1) * (theta / (2 * np.pi) + 0.5)
    y_2 = (height - 1) * (r / 2)

    return (x_2, y_2)


def compute_polar_angles(x_dest, y_dest, width_equirect, height_equirect):
    # -pi to pi
    theta = 2.0 * np.pi * (x_dest / width_equirect - 0.5)
    # -pi/2 to pi/2
    phi = np.pi * (y_dest / height_equirect - 0.5)
    return (theta, phi)

def vector_in_3D_space(theta, phi):
    x = np.cos(phi) * np.sin(theta)
    y = np.cos(phi) * np.cos(theta)
    z = np.sin(phi)
    return (x, y, z)

def calculate_fisheye_angle_radius(x, y, z, width, fov):
    theta = np.arctan2(z, x)
    phi = np.arctan2(np.sqrt(x**2 + z ** 2), y)
    r = width * phi / fov
    return (theta, phi, r)


def fish2sphere(x, y, width, height, fov, width_equirect, height_equirect):
    (theta_f, phi_f) = compute_polar_angles(x, y, width_equirect, height_equirect)
    (x_3d, y_3d, z_3d) = vector_in_3D_space(theta_f, phi_f)
    (theta_sphere, phi_sphere, r_sphere) = calculate_fisheye_angle_radius(x_3d, y_3d, z_3d, width, fov)

    # fisheye space
    x_pfish = 0.5 * width + r_sphere * np.cos(theta_sphere)
    y_pfish = 0.5 * width + r_sphere * np.sin(theta_sphere)

    return (x_pfish, y_pfish)

def crop_non_zero_region(image):
    # Find coordinates of non-zero pixels
    non_zero_coords = np.argwhere(np.any(image != 0, axis=-1))
    
    # Extract x and y coordinates
    x_coords = non_zero_coords[:, 1]
    y_coords = non_zero_coords[:, 0]
    
    # Find bounding box coordinates
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    
    # Crop the image using the bounding box coordinates
    cropped_image = image[min_y:max_y+1, min_x:max_x+1]
    
    return cropped_image

def project_image(input_image_path, output_image_path, fov):
    fisheye_image = cv2.imread(input_image_path)
    height, width = fisheye_image.shape[:2]

    width_equirect = width * 2
    height_equirect = height

    aperture = math.radians(fov)

    # Create a blank image for the unwrapped equirectangular projection
    unwrapped_image = np.zeros((height_equirect, width_equirect, 3), dtype=np.uint8)

    # Generate meshgrid of coordinates for the unwrapped image
    x_indices = np.arange(width_equirect)
    y_indices = np.arange(height_equirect)
    xx, yy = np.meshgrid(x_indices, y_indices)

    # Convert meshgrid coordinates to fish-eye coordinates
    x_pfish, y_pfish = fish2sphere(xx, yy, width, height, aperture, width_equirect, height_equirect)
    # print(x_pfish.shape)

    # Clip coordinates to ensure they fall within the bounds of the fisheye image
    x_pfish_clipped = np.clip(x_pfish, 0, width - 1)
    y_pfish_clipped = np.clip(y_pfish, 0, height - 1)

    # Map the pixel values from fisheye image to the unwrapped image
    unwrapped_image = fisheye_image[y_pfish_clipped.astype(int), x_pfish_clipped.astype(int)][:, width_equirect//4:3*width_equirect//4, :]
    
    # no_info_coords = np.argwhere(np.all(unwrapped_image == 0, axis=-1))
    # unwrapped_image =  crop_non_zero_region(unwrapped_image)
    # print(no_info_coords.shape)

    # Save the unwrapped image
    cv2.imwrite(output_image_path, unwrapped_image)
    return unwrapped_image

def project_image_loop(input_image_path, output_image_path, fov):
    fisheye_image = cv2.imread(input_image_path)
    height, width = fisheye_image.shape[:2]

    width_equirect = width * 2
    height_equirect = height

    aperture = np.radians(fov)
    l = []

    # Create a blank image for the unwrapped equirectangular projection
    unwrapped_image = np.zeros((height_equirect, width_equirect, 3), dtype=np.uint8)
    
    taken_points = np.zeros(width_equirect)

    for x in range(width_equirect):
        for y in range(height_equirect):
            (x_pfish, y_pfish) = fish2sphere(x, y, width, height, aperture, width_equirect, height_equirect)

            # Map the pixel value from fisheye image to the unwrapped image
            if 0 <= x_pfish < width and 0 <= y_pfish < height:
                taken_points[x] += 1;
                unwrapped_image[y, x] = fisheye_image[int(y_pfish), int(x_pfish)]
                
    full_image = np.argwhere(taken_points == height_equirect)
    start = full_image[0][0] - 5
    end = full_image[-1][0] + 5
    
    unwrapped_image = unwrapped_image[:, start+50:end-50, :]
    
    
    
    # Save the unwrapped image
    
    
    cv2.imwrite(output_image_path, unwrapped_image)
    return unwrapped_image

def split_image(input_image_path, output_image_path1, output_image_path2):
    # Open the image
    image = Image.open(input_image_path)
    print(f"new size: {image.size}")


    # Get the width and height of the image
    width, height = image.size

# Calculate new dimensions by dividing by 2


# Use the resize method to downsample the image
    # image = image.resize((width//2, height//2 ), Image.ANTIALIAS)
    # width, height = image.size

    # print(f"new size: {image.size}")
    # Calculate the midpoint
    midpoint = width // 2

    # Split the image into left and right halves
    left_half = image.crop((0, 0, midpoint, height))
    right_half = image.crop((midpoint, 0, width, height))

    left_half.save(output_image_path1)
    right_half.save(output_image_path2)
    return left_half, right_half


def crop_left_half(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the midpoint
    midpoint = width // 2

    # Crop the left half of the image
    left_half = image.crop((0, 0, midpoint, height))

    left_half.save(output_path)


def crop_right_half(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the midpoint
    midpoint = width // 2

    # Crop the left half of the image
    right_half = image.crop((midpoint, 0, width, height))

    right_half.save(output_path)


def crop_middle_half(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Get the width and height of the image
    width, height = image.size

    # Calculate the quarter points
    quarter_point = width // 4
    three_quarter_point = 3 * quarter_point

    # Crop the middle half of the image
    middle_half = image.crop((0, 0, width, height))

    middle_half.save(output_path)
    image_array = np.array(middle_half)

    return image_array


def unwrap_and_cut_img(input_image_path, output_left_path, output_right_path, fov):
    # Split the image into left and right halves
    left_path = "Images_cut/img_left_half.jpg"
    right_path = "Images_cut/img_right_half.jpg"
    left_half, right_half = split_image(input_image_path, left_path, right_path)

    left_final_path = "Images_undistorted/left_projected.jpg"
    right_final_path = "Images_undistorted/right_projected.jpg"
    unwrapped_left = project_image(left_path, output_left_path, fov)
    unwrapped_right = project_image(right_path, output_right_path, fov)

    # # # crop left one
    # left_img_array = crop_middle_half(left_final_path, output_left_path)
    # # crop right one
    # right_img_array = crop_middle_half(right_final_path, output_right_path)
    return unwrapped_left, unwrapped_right
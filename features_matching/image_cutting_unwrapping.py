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
    longitude = math.pi * x_prime + lambda_0
    latitude = math.pi/2 * y_prime
    return (longitude, latitude)


def normalized_2D_fisheye_coord(latitude, longitude, aperture_in_rad, width, height, x, y):
    P_x = math.cos(latitude) * math.cos(longitude)
    P_y = math.cos(latitude) * math.sin(longitude)
    P_z = math.sin(latitude)

    theta = math.atan2(P_z, P_x)

    sqrt_for_r = math.sqrt(P_x ** 2 + P_z ** 2)
    if (P_y != 0):
        r = 2 * math.atan2(sqrt_for_r, P_y) / aperture_in_rad
    else:
        r = 0

    x_2 = (width - 1) * (theta / (2 * math.pi) + 0.5)
    y_2 = (height - 1) * (r / 2)

    return (x_2, y_2)


def compute_polar_angles(x_dest, y_dest, width_equirect, height_equirect):
    # -pi to pi
    theta = 2.0 * math.pi * (x_dest / width_equirect - 0.5)
    # -pi/2 to pi/2
    phi = math.pi * (y_dest / height_equirect - 0.5)
    return (theta, phi)

def vector_in_3D_space(theta, phi):
    x = math.cos(phi) * math.sin(theta)
    y = math.cos(phi) * math.cos(theta)
    z = math.sin(phi)
    return (x, y, z)

def calculate_fisheye_angle_radius(x, y, z, width, fov):
    theta = math.atan2(z, x)
    phi = math.atan2(math.sqrt(x**2 + z ** 2), y)
    r = width * phi / fov
    return (theta, phi, r)


def fish2sphere(x, y, width, height, fov, width_equirect, height_equirect):
    (theta_f, phi_f) = compute_polar_angles(x, y, width_equirect, height_equirect)
    (x_3d, y_3d, z_3d) = vector_in_3D_space(theta_f, phi_f)
    (theta_sphere, phi_sphere, r_sphere) = calculate_fisheye_angle_radius(x_3d, y_3d, z_3d, width, fov)

    # fisheye space
    x_pfish = 0.5 * width + r_sphere * math.cos(theta_sphere)
    y_pfish = 0.5 * width + r_sphere * math.sin(theta_sphere)

    return (x_pfish, y_pfish)


def project_image(input_image_path, output_image_path, fov):
    fisheye_image = cv2.imread(input_image_path)
    height, width = fisheye_image.shape[:2]

    width_equirect = width * 2
    height_equirect = height

    aperture = math.radians(fov)
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
    print(start, end)
    # print(np.max())
    unwrapped_image = unwrapped_image[:, start:end, :]
    
    
    
    # Save the unwrapped image
    
    
    cv2.imwrite(output_image_path, unwrapped_image)
    return unwrapped_image

def split_image(input_image_path, output_image_path1, output_image_path2):
    # Open the image
    image = Image.open(input_image_path)

    # Get the width and height of the image
    width, height = image.size

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
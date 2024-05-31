# ComputationalPhoto

The project contains the main script `stitch.py` that performs stitching of images and video. The code for the unwrapping is included in `image_cutting_unwrapping.py`. Finally, the project also contains a tranform matrix saved in a .npy file `tr_matrix.py`. This matrix precomputed the "best" transformation matrix for stitching two images for the Xiaomi Mi Sphere 360.

## Usage

### Installation

Install the packages:
pip install -r requirements.txt

### Image stitching

For stitching multiple images without using the precomputed matrix:
python stitch.py image1.jpg image2.jpg...

For stitching multiple images with the precomputed matrix:
python stitch.py -p image1.jpg image2.jpg...

### Video stitching

For stitching a video without using the precomputed matrix:
python stitch.py -v video.mp4

For stitching a video with the precomputed matrix:
python stitch.py -p -v video.mp4

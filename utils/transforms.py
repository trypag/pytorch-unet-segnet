import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates


def resize_2d(image, size, interp=cv2.INTER_CUBIC):
    """Resize an image to the desired size with a specific interpolator.

    image: (ndarray) 3d ndarray of size CxHxW.
    size: (tuple) output size of the image.
    interp: (int) OpenCV interpolation type.
    """
    return cv2.resize(image.swapaxes(0, 2), size,
                      interpolation=interp).swapaxes(0, 2)


def center_rotate(image, angle, scale=1., interp=cv2.INTER_LANCZOS4):
    """Apply a center rotation on a 2d image with a defined angle and scale.
    It supports 2d images with multiple channels.

    image: (ndarray) 3d ndarray of size CxHxW.
    angle: (float) rotation angle in degrees.
    scale: (float) scaling factor.
    interp: (int) OpenCV interpolation type.
    """
    h = image.shape[1]
    w = image.shape[2]

    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((h // 2, w // 2), angle, scale)
    # apply the affine transform on the image
    return cv2.warpAffine(image.swapaxes(0, 2), rot_mat, (h, w),
                          flags=interp).swapaxes(0, 2)


def elastic_deformation_2d(image, alpha, sigma, order=1, mode='constant',
                           constant=0, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_
    Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural
    Networks applied to Visual Document Analysis"
    Based on https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

    image: (ndarray) 3d ndarray of size CxHxW.
    alpha: (number) Intensity of the deformation.
    sigma: (number) Sigma for smoothing the transformation.
    order: (int) coordinate remapping : order of the spline interpolation.
    mode: (str) coordinate remapping : interpolation type.
    constant: (int) constant value if mode is 'constant'.
    random_state: (RandomState) Numpy random state.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # random displacement field
    def_x = random_state.rand(*shape[-2:]) * 2 - 1
    def_y = random_state.rand(*shape[-2:]) * 2 - 1

    # smooth the displacement field of x,y axis
    dx = cv2.GaussianBlur(def_x, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(def_y, (0, 0), sigma) * alpha

    # repeat the displacement field for each channel
    dx = np.repeat(dx[np.newaxis, :], shape[0], axis=0)
    dy = np.repeat(dy[np.newaxis, :], shape[0], axis=0)

    # grid of coordinates
    x, z, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                          np.arange(shape[2]))

    indices = (z.reshape(-1, 1), np.reshape(x + dx, (-1, 1)),
               np.reshape(y + dy, (-1, 1)))

    return map_coordinates(image, indices, order=order,
                           mode=mode).reshape(shape)


def elastic_deformation_3d(image, alpha, sigma, order=1, mode='constant',
                           constant=0, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_
    Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural
    Networks applied to Visual Document Analysis"
    Based on https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

    image: (ndarray) 3d ndarray of size DxHxW.
    alpha: (number) Intensity of the deformation.
    sigma: (number) Sigma for smoothing the transformation.
    order: (int) coordinate remapping : order of the spline interpolation.
    mode: (str) coordinate remapping : interpolation type.
    constant: (int) constant value if mode is 'constant'.
    random_state: (RandomState) Numpy random state.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # random displacement field
    def_x = random_state.rand(*shape) * 2 - 1
    def_y = random_state.rand(*shape) * 2 - 1
    def_z = random_state.rand(*shape) * 2 - 1

    # smooth the displacement field of each axis
    dx = cv2.GaussianBlur(def_x, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(def_y, (0, 0), sigma) * alpha
    dz = cv2.GaussianBlur(def_z, (0, 0), sigma) * alpha

    # grid of coordinates
    x, z, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                          np.arange(shape[2]))

    indices = ((z + dz).reshape(-1, 1), np.reshape(x + dx, (-1, 1)),
               np.reshape(y + dy, (-1, 1)))

    return map_coordinates(image, indices, order=order,
                           mode=mode).reshape(shape)

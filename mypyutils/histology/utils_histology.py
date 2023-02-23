
# region packages
import os.path as op
from colorama import Fore, Style
import itertools

import numpy as np
from matplotlib import pyplot as plt

import torch.nn
import skimage

import cv2
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

from ..utils_general import *

# endregion
################################################


def image_type(image, method='otsu', otsu_thresh=None):
    m = image.mean()
    m1 = image[:, :, 0].mean()
    m2 = image[:, :, 1].mean()
    m3 = image[:, :, 2].mean()

    img_type = None
    check_mean_channels = lambda f: f * m < m1 < (2 - f) * m and f * m < m2 < (2 - f) * m and f * m < m3 < (2 - f) * m
    if check_mean_channels(0.95):  # blank - partial
        if not check_mean_channels(0.975):
            img_type = 'partial'
        else:
            img_type = 'blank'

    if image.ndim == 3 and image.shape[0] == 3:
        gray_img = np.mean(image, axis=0)
    elif image.ndim == 3 and image.shape[-1] == 3:
        gray_img = np.mean(image, axis=-1)
    else:
        gray_img = image

    if method == 'otsu' and img_type is None:
        if otsu_thresh is None:
            otsu_thresh = skimage.filters.threshold_otsu(image=gray_img, nbins=50)
        tissue_perc = np.mean(gray_img <= otsu_thresh) * 100
        if tissue_perc >= 50:
            img_type = 'tissue'
        elif 20 <= tissue_perc < 50:
            img_type = 'partial'
        else:
            img_type = 'blank'

    if method == 'average' and img_type is None:
        if m > 200:
            if m1 < 200 or m2 < 200 or m3 < 200:
                img_type = 'tissue'
            else:
                img_type = 'partial'
        else:
            img_type = 'tissue'
    return img_type
################################################


class HistologyTiles(DeepZoomGenerator):
    def __init__(self, osr, tile_size=254, overlap=1, limit_bounds=False):
        super().__init__(osr, tile_size, overlap, limit_bounds)

    @property
    def native_level_num(self):
        tile_native_level = self.level_count
        return tile_native_level

    def number_of_patches(self, level_num):
        dim1, dim2 = self.level_tiles[level_num]
        return dim1, dim2

    def patch_type(self, d1, d2, method='average', otsu_thresh=None):
        tile_native_level = self.level_count
        level_num = tile_native_level - 1
        tile_d1_d2 = np.array(self.get_tile(level_num, (d1, d2)))

        img_type = image_type(tile_d1_d2, method=method, otsu_thresh=otsu_thresh)
        return img_type, tile_d1_d2

    def save_patches(self, save_patch_types, dim1_range, dim2_range, samp_name, path_data_tiles,
                     method='average', otsu_thresh=None):
        for d1 in dim1_range:
            for d2 in dim2_range:
                img_type, img_patch = self.patch_type(d1, d2, method=method, otsu_thresh=otsu_thresh)
                if img_type in save_patch_types:
                    tile_name = combine_names('_', samp_name, d1, d2)
                    tile_dir = op.join(path_data_tiles + '-' + img_type, tile_name)
                    plt.imsave(tile_dir + '.png', img_patch)


################################################
################################################
def norm_HnE(img, Io=240, alpha=1, beta=0.15):
    """
    method based on:
    http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    code from : https://github.com/bnsreenu/python_for_microscopists
    :param img:
    :param Io:
    :param alpha:
    :param beta:
    :return:
    """
    ######## Step 1: Convert RGB to OD ###################
    ## reference H&E OD matrix.
    # Can be updated if you know the best values for your image.
    # Otherwise use the following default values.
    # Read the above referenced papers on this topic.
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    ### reference maximum stain concentrations for H&E
    maxCRef = np.array([1.9705, 1.0308])

    # extract the height, width and num of channels of image
    h, w, c = img.shape

    # reshape image to multiple rows and 3 columns.
    # Num of rows depends on the image size (wxh)
    img = img.reshape((-1, 3))

    # calculate optical density
    # OD = −log10(I)
    # OD = -np.log10(img+0.004)  #Use this when reading images with skimage
    # Adding 0.004 just to avoid log of zero.

    OD = -np.log10((img.astype(float) + 1) / Io)  # Use this for opencv imread
    # Add 1 in case any pixels in the image have a value of 0 (log 0 is indeterminate)

    ############ Step 2: Remove data with OD intensity less than β ############
    # remove transparent pixels (clear region with no tissue)
    ODhat = OD[~np.any(OD < beta, axis=1)]  # Returns an array where OD values are above beta
    # Check by printing ODhat.min()

    ############# Step 3: Calculate SVD on the OD tuples ######################
    # Estimate covariance matrix of ODhat (transposed)
    # and then compute eigen values & eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    ######## Step 4: Create plane from the SVD directions with two largest values ######
    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])  # Dot product

    ############### Step 5: Project data onto the plane, and normalize to unit length ###########
    ############## Step 6: Calculate angle of each point wrt the first SVD direction ########
    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T

    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    ###### Step 8: Convert extreme values back to OD space
    # recreate the normalized image using reference mixing matrix

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # Separating H and E components

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return (Inorm, H, E)


class NormHnE(object):
    def __init__(self, Io=240, alpha=1, beta=0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        try:
            im_norm, _, _ = norm_HnE(np.array(img), Io=self.Io, alpha=self.alpha, beta=self.beta)
        except:
            print(Fore.LIGHTBLUE_EX + 'Warning: H&E normalization failed, the original image is returned!')
            print(Style.RESET_ALL)
            im_norm = np.array(img)
        return Image.fromarray(np.uint8(im_norm))

    # def __repr__(self) -> str:
    #     return f"{self.__class__.__name__}(Io={self.Io}, alpha={self.alpha}, beta={self.beta})"

################################################
################################################


class HistologyWSI(openslide.OpenSlide):
    def __init__(self, filename):
        super().__init__(filename)


################################################
################################################
def otsu_segment_wsi(slide):
    slide_thumb_600 = slide.get_thumbnail(size=(600, 600))  # an image of size slide_thumb_600.size
    slide_thumb_600_np = np.array(slide_thumb_600)
    thresh_otsu = skimage.filters.threshold_otsu(image=np.mean(slide_thumb_600_np, axis=-1))
    return thresh_otsu

################################################
################################################
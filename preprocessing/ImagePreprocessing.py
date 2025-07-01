'''
Image Preprocessing Utilities

1. Bias Field Correction

2. Denoising 

3. Sharpening

4. Intensity Standardization

'''

import os
import tempfile
import shutil
import gc

import math

import numpy as np
import SimpleITK as sitk

from scipy.interpolate import interp1d
from sklearn.utils.extmath import cartesian
from skimage.exposure import rescale_intensity

#from ants import ants_image_io, ants_image
#from ants.utils import n3_bias_field_correction

from dipy.denoise import noise_estimate, nlmeans

from numba import jit, prange
from joblib import Parallel, delayed, dump, load



# ---------------- #
# Median Filtering #
# ---------------- #
def median_filter(image, radius = None):
    '''
    Median filter
    
    '''
    med_filt = sitk.MedianImageFilter()

    if radius is not None:

        if hasattr(radius, '__len__'):
            assert len(radius) == 3, 'Incorrect number of elements'
        med_filt.SetRadius(list(radius))

    else:

        image = med_filt.Execute(image)

    return image
            

def laplacian_sharpening(image):
    '''
    Laplacian Sharpening image filter

    '''

    lap_sharp_filter = sitk.LaplacianSharpeningImageFilter()
    image = lap_sharp_filter.Execute(image)

    return image


def curvature_flow(image, time_step = 0.02, n_iter = 10):
    '''
    Curvature flow diffusion image filter
    
    '''

    n_dim = image.GetDimension()
    spacing = image.GetSpacing()
    assert time_step < min(spacing) / (2**(n_dim + 1)), 'TimeStep too large'

    curv_filt = sitk.CurvatureFlowImageFilter()
    curv_filt.SetTimeStep(time_step)
    curv_filt.SetNumberOfIterations(n_iter)

    image = curv_filt.Execute(image)

    return image


def gradient_diffusion(image, time_step = 0.02, conductance_param = 1.0, n_iter = 10):
    '''
    Anisotropic gradient diffusion image filter
    
    '''

    n_dim = image.GetDimension()
    spacing = image.GetSpacing()
    assert time_step < min(spacing) / (2**(n_dim + 1)), 'TimeStep too large'

    grad_diff_filt = sitk.GradientAnisotropicDiffusionImageFilter()
    grad_diff_filt.SetTimeStep(time_step)
    grad_diff_filt.SetConductanceParameter(conductance_param)
    grad_diff_filt.SetNumberOfIterations(n_iter)

    image = grad_diff_filt.Execute(sitk.Cast(image, sitk.sitkFloat32))

    return image


def bias_field_correction(image, downsample_factor = 4):
    '''
    N3 bias field correction filter

    '''

    assert type(image) == sitk.SimpleITK.Image, 'Incorrect image data type'

    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    ants_img = ants_image_io.from_numpy(sitk.GetArrayFromImage(image).astype('float'))
    ants_img = n3_bias_field_correction(ants_img, downsample_factor = downsample_factor)

    image = sitk.GetImageFromArray(ants_img.numpy())
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def histogram_matching(image, template, n_points = 3, n_levels = 100):
    '''
    Histogram matching image filter

    '''

    assert type(image) == sitk.SimpleITK.Image and type(template) == sitk.SimpleITK.Image, 'Incorrect image type'

    hist_match = sitk.HistogramMatchingImageFilter()
    hist_match.SetNumberOfMatchPoints(n_points)
    hist_match.SetNumberOfHistogramLevels(n_levels)

    image = hist_match.Execute(image, template)

    return image


def rescale(image, min_value = 0, max_value = 100):
	'''
	Rescale image intensities to [min_value, max_value]
	'''

	if type(image) == sitk.SimpleITK.Image:

		rescaler = sitk.RescaleIntensityImageFilter()
		rescaler.SetOutputMinimum(min_value)
		rescaler.SetOutputMaximum(max_value)
		image = rescaler.Execute(image)
	
	else:

		image = rescale_intensity(image, out_range = (min_value, max_value))

	return image



class IntensityStandardizer:
    '''
    Nyul Intensity Standardization class

    Training phase:
        Learns histogram modes from set of training images

    Transformation phase:
        Piece-wise linearly matches histogram of test image to templated histogram

    ...

    Attributes
    ----------

    Methods
    -------

    '''

    def __init__(self, numPoints, sMin = 1, sMax = 100, pLow = 1, pHigh = 99):
        '''
        Initialize class variables
        '''

        self.numPoints = numPoints
        self.sMin = sMin
        self.sMax = sMax
        self.pLow = pLow
        self.pHigh = pHigh
        self.perc = np.asarray([pLow] + list(np.arange(0, 100, numPoints)[1:]) + [pHigh])
    

    def getLandmarks(self, image, mask):
        '''
        computes image landmarks
        '''

        image_nda = sitk.GetArrayFromImage(image)
        mask_nda = sitk.GetArrayFromImage(mask)

        lms = [np.percentile(image_nda[mask_nda > 0], i) for i in self.perc]
        mapping = interp1d([lms[0], lms[-1]], [self.sMin, self.sMax], fill_value = 'extrapolate')
        mapped_lms = mapping(lms)

        return mapped_lms


    def train(self, imageList, maskList):
        '''
        Computes landmarks for all images
        '''

        self.mean_landmarks = []
        mapped_landmarks = []

        for image_path, mask_path in zip(imageList, maskList):
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)

            mapped_lms = self.getLandmarks(image, mask)
            mapped_landmarks.append(mapped_lms)

        mapped_landmarks = np.asarray(mapped_landmarks)
        self.mean_landmarks = mapped_landmarks.mean(axis = 0)


    def transform(self, image, mask):
        '''
        Standardizes image histogram based on computed landmarks
        '''

        lms = self.getLandmarks(image, mask)
        mapping = interp1d(lms, self.mean_landmarks, fill_value = 'extrapolate')
        
        image_nda = sitk.GetArrayFromImage(image)
        mapped_image_nda = mapping(image_nda.ravel())
        mapped_image_nda = mapped_image_nda.reshape(image_nda.shape)
        
        mapped_image = sitk.GetImageFromArray(mapped_image_nda)
        mapped_image.CopyInformation(image)

        return mapped_image


    def saveModel(self, path):
        '''
        saves computed landmarks and model parameters
        '''

        model = {'pLow': self.pLow,
                 'pHigh': self.pHigh,
                 'sMin': self.sMin,
                 'sMax': self.sMax,
                 'landmarks': self.mean_landmarks}

        np.savez(path, model)


    def loadModel(self, path):
        '''
        loads computed landmarks and model parameters
        '''

        model = np.load(path, allow_pickle = True)
        return model



class BlockNonLocalMeans:
    '''
    Block-wise Non-Local Means Denoising (Coupe et al 2008)
    Optimized & parallelized version of the classic non-local means denoising algorithm

    ...

    Attributes
    ----------
    M : int
        the search volume size (2M + 1)^3
    a : int
        the block size (2a + 1)^3
    ngird: int
        spacing between central voxels of each block
    mu1 : float (optional) 
        parameter used for block selection - speed up purposes
    sigma1 : float (optional)
        parameter used for block selection - speed up purposes
    Methods
    -------

    '''

    def __init__(self, M, a, ngrid, mu1, sigma1):
        self.M = M
        self.a = a
        self.ngrid = ngrid
        self.mu1 = mu1
        self.sigma1 = sigma1


    def EstimateNoiseLevel(self, image):
        
        return self._EstimateNoiseLevel(image)


    
    @staticmethod
    @jit(parallel = True)
    def _EstimateNoiseLevel(image):
        
        Xrange = np.arange(1, image.shape[0] - 1)
        Yrange = np.arange(1, image.shape[1] - 1)
        Zrange = np.arange(1, image.shape[2] - 1) 
        voxels = cartesian((Xrange, Yrange, Zrange))
        n_voxels = voxels.shape[0]

        #to ensure uniformity of noise variance in homogeneous regions
        coef = math.sqrt(6 / 7)

        #estimation
        epsilon = np.zeros(shape = (n_voxels, ))
        for i in prange(n_voxels):
            x, y, z = voxels[i]
            neighbors = np.array([image[x - 1, y, z],
                                  image[x + 1, y, z],
                                  image[x, y - 1, z],
                                  image[x, y + 1, z],
                                  image[x, y, z - 1],
                                  image[x, y, z + 1]])
            epsilon[i] = coef * (image[x, y, z] - neighbors.mean())
        
        sigma_noise = (1 / n_voxels) * (epsilon**2).sum()

        return sigma_noise


    def ApplyFilter(self, image, beta):
        '''
        Apply non-local means to block centers in 1st stage
        Refine each voxel according to blocks in which it resides
        '''

        #check image type
        if type(image) == sitk.SimpleITK.Image:
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            image = sitk.GetImageFromArray(image)
            recast = True
        else:
            recast = False

        #check overlapping of block spacing and block size
        assert 2 * self.a >= self.ngrid, 'Block size and block spacing not compatible'

        #convert to float for numerical reasons
        if image.dtype is not 'float32':
            image = image.astype('float32')

        shape = image.shape
        rows, cols, slices = shape

        Xgrid = np.arange(self.M, self.M + rows, self.ngrid)
        Ygrid = np.arange(self.M, self.M + cols, self.ngrid)
        Zgrid = np.arange(self.M, self.M + slices, self.ngrid)
        block_centers = cartesian((Xgrid, Ygrid, Zgrid))

        #pad image and memmap
        image_padded = np.pad(image, pad_width = self.M, mode = 'reflect')
        shared_path = tempfile.mkdtemp()
        filename = os.path.join(shared_path, 'shared_image.mmap')
        image_padded_shared = np.memmap(filename, dtype = image_padded.dtype, mode = 'w+', shape = image_padded.shape)
        image_padded_shared[:] = image_padded[:]

        #create shared array to be writable by worker processes - output image
        output_filename = os.path.join(shared_path, 'output_shared_image.mmap')
        output_image_shared = np.memmap(output_filename, dtype = image_padded.dtype, mode = 'w+', shape = image_padded.shape)

        #create shared array to be writable by worker processes - overlap map
        overlap_filename = os.path.join(shared_path, 'overlap_map.mmap')
        overlap_map = np.ones(shape = image_padded.shape)
        overlap_map_shared = np.memmap(overlap_filename, dtype = overlap_map.dtype, mode = 'w+', shape = image_padded.shape)
        overlap_map_shared[:] = overlap_map[:]

        #Noise variance estimation
        sigma_noise = self.EstimateNoiseLevel(image)
        self.sigma = 2 * beta * sigma_noise * (2*self.a + 1)**3

        #block-wise nonlocal means
        Parallel(n_jobs = -1, max_nbytes = None)(delayed(self.ApplyFilter_worker)(point, image_padded_shared, output_image_shared, overlap_map_shared) for point in block_centers)
        output_image = np.array(output_image_shared)
        overlap_map = np.array(overlap_map_shared)
        output_image = output_image / overlap_map
        output_image = output_image[self.M : self.M + rows,
                                    self.M : self.M + cols,
                                    self.M : self.M + slices]

        #clean up
        shutil.rmtree(shared_path)
        del image_padded_shared, output_image_shared, overlap_map_shared
        gc.collect()

        #recast if necessary
        if recast is True:
            output_image = sitk.GetImageFromArray(output_image)
            output_image.SetOrigin(origin)
            output_image.SetSpacing(spacing)
            output_image.SetDirection(direction)

        return output_image


    def ApplyFilter_worker(self, point, image, output_image, overlap_map):
        '''
        Apply non local means adaptive filtering
        to whole blocks of image
        '''

        x, y, z = point

        #Get block centers within search volume
        #** slicing and adding 1 to align indices
        xg = np.arange(x - self.M, x + self.M + 1, self.ngrid)[:-1] + 1
        yg = np.arange(y - self.M, y + self.M + 1, self.ngrid)[:-1] + 1
        zg = np.arange(z - self.M, z + self.M + 1, self.ngrid)[:-1] + 1
        in_volume_block_centers = cartesian((xg, yg, zg))
        in_volume_block_centers = np.delete(in_volume_block_centers, (self.M**3) // 2, axis = 0)

        #set up overlap mask to account for overlapping blocks
        #TODO : mask only works for block size 3x3x3, make block-size invariant
        overlap_mask = np.array([[[8, 4, 8],
                                  [4, 2, 4],
                                  [8, 4, 8]],
                                 [[4, 2, 4],
                                  [2, 1, 2],
                                  [4, 2, 4]],
                                 [[8, 4, 8],
                                  [4, 2, 4],
                                  [8, 4, 8]]])
        overlap_map[x - self.a : x + self.a + 1,
                    y - self.a : y + self.a + 1,
                    z - self.a : z + self.a + 1] = overlap_mask

        #get block-wise intensity vectors
        Bi = image[x - self.a : x + self.a + 1,
                   y - self.a : y + self.a + 1,
                   z - self.a : z + self.a + 1]
        uBi = Bi.ravel()
        uBj = []
        w = []

        #calculate block weight-coefficients
        for i in range(self.M**3 - 1):
            xi, yi, zi = in_volume_block_centers[i]
            bj = image[xi - self.a : xi + self.a + 1,
                       yi - self.a : yi + self.a + 1,
                       zi - self.a : zi + self.a + 1]
            ubj = bj.flatten()
            wj = np.exp(-(np.linalg.norm(uBi - ubj)**2) / self.sigma)
            w.append(wj)
            uBj.append(ubj)
        w = np.array(w)
        w = w / w.sum()

        #get reconstructed block as weighted average of in-search-volume blocks
        reconstructed_block = w.dot(np.array(uBj))
        reconstructed_block = reconstructed_block.reshape([2*self.a + 1]*3)
        
        #append block to output image
        output_image[x - self.a : x + self.a + 1,
                     y - self.a : y + self.a + 1,
                     z - self.a : z + self.a + 1] += reconstructed_block



class AdaptiveMedianFilter:
    '''
    Patch-based denoising median filter. 
    Effective against impulse noise.
    Central voxel intensity treated according to whether is corresponds to impulse response or not

    ...

    Attributes
    ----------
    S : int or array-like
        initial window size, can be different for each axis
    Smax : int or array-like
        maximum allower window size, can be different for each axis

    Methods
    -------
    ApplyFilter(image)
        Initializes filtering process and distributes workload to worker processes
    ApplyFilter_worker(point, image)
        Applied adaptive median filtering to specific voxel

    '''

    def __init__(self, S, Smax):
        self.S = S
        self.Smax = Smax

    
    def ApplyFilter(self, image):
        '''
        Apply filtering process
        '''

        #check image format
        if type(image) == sitk.SimpleITK.Image:
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()
            image = sitk.GetArrayFromImage(image)
            recast = True
        else:
            recast = False

        #convert image to float data type
        image = image.astype('float32')

        shape = image.shape
        rows, cols, slices = shape

        #check for isotropic / anisotropic window size
        if isinstance(self.S, (list, tuple, np.ndarray)):
            self.sx, self.sy, self.sz = self.S
            self.sx_max, self.sy_max, self.sz_max = self.Smax
        else:
            self.sx, self.sy, self.sz = self.S, self.S, self.S
            self.sx_max, self.sy_max, self.sz_max = self.Smax, self.Smax, self.Smax

        Xrange = np.arange(self.sx_max, rows + self.sx_max)
        Yrange = np.arange(self.sy_max, cols + self.sy_max)
        Zrange = np.arange(self.sz_max, slices + self.sz_max)
        points = cartesian((Xrange, Yrange, Zrange))

        #memmap padded image to share between processes
        image_padded = np.pad(image, pad_width = ((self.sx_max, self.sx_max), (self.sy_max, self.sy_max), (self.sz_max, self.sz_max)), mode = 'edge')
        temp_path = tempfile.mkdtemp()
        filename = os.path.join(temp_path, 'shared_image.mmap')
        image_padded_shared = np.memmap(filename, dtype = image_padded.dtype, mode = 'w+', shape = image_padded.shape)
        image_padded_shared[:] = image_padded[:]

        #Distribute workload to workers
        filtered_image = Parallel(n_jobs = -1, max_nbytes = None)(delayed(self.ApplyFilter_worker)(point, image_padded_shared) for point in points)
        filtered_image = np.array(filtered_image).reshape(shape)

        #clean up
        del image_padded_shared
        shutil.rmtree(temp_path)
        gc.collect()

        if recast is True:
            filtered_image = sitk.GetImageFromArray(filtered_image)
            filtered_image.SetOrigin(origin)
            filtered_image.SetSpacing(spacing)
            filtered_image.SetDirection(direction)

        return filtered_image


    def ApplyFilter_worker(self, point, image):
        '''
        Worker process, multiple instances running on multiple CPUs
        '''

        x, y, z = point
        rx, ry, rz = self.sx, self.sy, self.sz

        while(True):
            patch = image[x - rx : x + rx + 1,
                          y - ry : y + ry + 1,
                          z - rz : z + rz + 1]
            patch = patch.ravel()

            Imin = patch.min()
            Imax = patch.max()
            Imed = np.median(patch)
            Ixyz = image[x, y, z]

            if Imin < Imed and Imed < Imax:
                if Imin < Ixyz and Ixyz < Imax:
                    #not impulse
                    return Ixyz
                else:
                    return Imed
            else:
                rx += 1
                ry += 1
                rz += 1
                if rx == self.sx_max or ry == self.sy_max or rz == self.sz_max:
                    return Imed
                else:
                    continue

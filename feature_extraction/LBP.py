'''
-------------------------------------------------------------------
3D - LBP feature extraction

1. Fully 3D LBP extension

2. Triplanar (LBP features in three orthogonal planes [xy, yz, xz])
-------------------------------------------------------------------
'''

import math
import itertools

import numpy as np

import networkx as nx

from scipy.special import sph_harm
from scipy.ndimage.interpolation import map_coordinates
from scipy.stats import kurtosis

from trimesh.creation import icosphere

from sklearn.utils.extmath import cartesian
from sklearn.preprocessing import normalize

from skimage.feature import local_binary_pattern



class LBP_3D:
    '''
    3D LBP feature extraction by sampling 3x3x3 cube of adjacent and diagonal voxels - 
    6 & 8 neighbors respectively
    Reference: Supervoxel Clustering with a Novel 3D Descriptor for Brain Tissue Segmentation, 
               Liu Y., Du S., Kong Y., IJMLC, Vol. 10, No. 3, May 2020,
               doi: 10.18178/ijmlc.2020.10.3.964
               
    ...

    Attributes
    ----------
    patch_size : int
        size of cubic area centered around voxel to be characterized
    adjacent_bins : int
        number of bins for histogram of adjacent neighbors
    diagonal_bins : int
        number of bins for histogram of diagonal neighbors
    normalize : bool
        whether to normalize final descriptor or not

    Methods
    ------
    getDiagonalCode()
    getAdjacentCode()
    encode()

    '''

    def __init__(self, patch_size, adjacent_bins = 6, diagonal_bins = 8, normalize = False):
        '''
        Initialization
        '''

        self.patch_size = patch_size
        self.adjacent_bins = adjacent_bins
        self.diagonal_bins = diagonal_bins
        self.normalize = normalize


    def getDiagonalCode(self, point, image):
        '''
        Return texture code extracted from diagonal neighbors
        '''

        x, y, z = point
        central_val = image[x, y, z]

        diagonal_vals = []
        diagonal_vals.append(image[x - 1, y - 1, z - 1])
        diagonal_vals.append(image[x - 1, y - 1, z + 1])
        diagonal_vals.append(image[x - 1, y + 1, z - 1])
        diagonal_vals.append(image[x - 1, y + 1, z + 1])
        diagonal_vals.append(image[x + 1, y - 1, z - 1])
        diagonal_vals.append(image[x + 1, y - 1, z + 1])
        diagonal_vals.append(image[x + 1, y + 1, z - 1])
        diagonal_vals.append(image[x + 1, y + 1, z + 1])
        diagonal_vals = np.asarray(diagonal_vals)

        diagonal_code = diagonal_vals[diagonal_vals > central_val].size
        return 2**diagonal_code - 1


    def getAdjacentCode(self, point, image):
        '''
        Return texture code extracted from adjacent neighbors
        '''
        
        x, y, z = point
        central_val = image[x, y, z]

        adjacent_vals = []
        adjacent_vals.append(image[x - 1, y, z])
        adjacent_vals.append(image[x + 1, y, z])
        adjacent_vals.append(image[x, y - 1, z])
        adjacent_vals.append(image[x, y + 1, z])
        adjacent_vals.append(image[x, y, z - 1])
        adjacent_vals.append(image[x, y, z + 1])
        adjacent_vals = np.asarray(adjacent_vals)

        adjacent_code = adjacent_vals[adjacent_vals > central_val].size
        return 2**adjacent_code - 1

    
    def encode(self, points, image):
        '''
        Extract LBP features
        '''

        Descriptors = []

        for point in points:

            x, y, z = point
            step = self.patch_size // 2
            x_range = np.arange(x - step, x + step + 1)
            y_range = np.arange(y - step, y + step + 1)
            z_range = np.arange(z - step, z + step + 1)
            patch_points = cartesian((x_range, y_range, z_range))

            adjacent = []
            diagonal = []
            for patch_point in patch_points:
                adjacent.append(self.getAdjacentCode(patch_point, image))
                diagonal.append(self.getDiagonalCode(patch_point, image))

            adjacent = np.asarray(adjacent)
            diagonal = np.asarray(diagonal)

            adjacent_hist = np.histogram(adjacent, bins = self.adjacent_bins)[0]
            diagonal_hist = np.histogram(diagonal, bins = self.diagonal_bins)[0]

            if self.normalize:
                adjacent_hist = adjacent_hist / np.linalg.norm(adjacent_hist)
                diagonal_hist = diagonal_hist / np.linalg.norm(diagonal_hist)
            d = np.hstack((adjacent_hist, diagonal_hist))
            Descriptors.append(d)

        Descriptors = np.vstack(Descriptors)
        return Descriptors



def triplanar_lbp(point, image, patch_size, p, r, method = 'var', nbins = 10, normalize = True):
    '''
    LBP feature extraction on three orhogonal planes (xy-, yz-, xz-)
    Three separate LBP histograms are calculated and then concatenated to form the final descriptor

    ...

    Parameters
    ----------
    point : tuple 
        the voxel to be characterized
    image : ndarray
        MR image
    patch_size : int
        size of square patch centered around voxel
    p : int
        number of neighboring voxels to consider
    r : float
        radius of circle on which neighbors will be sampled
    method : str
        method to form the pattern
    nbins : int
        number of histogram bins for each of the three histograms
    normalize : bool
        whether to normalize final descriptor or not

    Returns
    -------
    D : ndarray

    '''

    x, y, z = point

    img_xy = image[:, :, z]
    img_yz = image[x, :, :]
    img_xz = image[:, y, :]
            
    step = patch_size // 2
    patch_xy = img_xy[x - step -r : x + step + r + 1,
                      y - step -r : y + step + r + 1]
                           
    patch_xz = img_xz[x - step - r : x + step + r + 1,
                      z - step - r : z + step + r + 1]
            
    patch_yz = img_yz[y - step - r : y + step + r + 1,
                      z - step - r : z + step + r + 1]

    lbp_xy = local_binary_pattern(patch_xy, P = p, R = r, method = method)
    lbp_yz = local_binary_pattern(patch_yz, P = p, R = r, method = method)
    lbp_xz = local_binary_pattern(patch_xz, P = p, R = r, method = method)

    
    hist_xy = np.histogram(lbp_xy[r + step : 2*(r + step) + 1], bins = nbins)[0]
    hist_yz = np.histogram(lbp_yz[r + step : 2*(r + step) + 1], bins = nbins)[0]
    hist_xz = np.histogram(lbp_xz[r + step : 2*(r + step) + 1], bins = nbins)[0]
    
    hist_xy = hist_xy / np.linalg.norm(hist_xy)
    hist_yz = hist_yz / np.linalg.norm(hist_yz)
    hist_xz = hist_xz / np.linalg.norm(hist_xz)

    D = np.hstack((hist_xy, hist_yz, hist_xz))

    if normalize is True:
        D = D / np.linalg.norm(D)

    return D



# -------------------------------------------------------------------------- #
# 3D - LBP Imlementation : Rotational Invariance through Spherical Harmonics #
# -------------------------------------------------------------------------- #
def lbp_ri_sh(point, img, patch_size, sph_degree, ico_radius, ico_level, n_bins = 20, concatenate = True):
    '''
    Computes a 3D LBP texture descriptor for a region centered around a voxel. The intensity values 
    of the neighboring voxels is treated a spherical function, and decomposed into a sum of
    spherical harmonics, achieving rotational invariance. A histogram of the texture codes is computed for
    each frequency (band) and the final descriptor is the concatenation of the above histograms.

    Reference : 3D LBP-Based Rotationally Invariant Region Description, 
                Banerjee J., Moelker A., Niessen W., Walsum  v. T., 
                ACCV 2012 Workshops, Part I, LNCS 7728, pp. 26-37, 2013

    ...

    Parameters
    ----------
    point : tuple
        the point to be described
    img : ndarray (width x height x depth)
        the MR image
    patch_size : int
        size of cellular patch around point
    sph_degree : int
        degree up to which to expand to spherical harmonics
    ico_radius : float
        radius of icosahedron to uniformly sample intensities around patch voxels
    ico_level : int
        controls refinement level of icosahedron
    n_bins : int
        number of bins for LBP histograms
    concatenate : bool
        if true, concatenate histograms, otherwise weighted aggregation

    Returns
    ------- 
    D : ndarray 
        LBP descriptor

    '''

    #extract image patch
    psize = patch_size // 2
    x, y, z = point
    patch = img[x - psize - int(math.ceil(ico_radius)) : x + psize + int(math.ceil(ico_radius)) + 1,
                y - psize - int(math.ceil(ico_radius)) : y + psize + int(math.ceil(ico_radius)) + 1,
                z - psize - int(math.ceil(ico_radius)) : z + psize + int(math.ceil(ico_radius)) + 1]
    
    patch_coords = cartesian((range(patch_size), range(patch_size), range(patch_size))) + 1    

    #construct icosahedron for uniform sampling on sphere surface
    ico = icosphere(subdivisions = ico_level, radius = ico_radius)
    ico_coords = np.array(ico.vertices)
    theta = np.arccos(ico_coords[:, 2] / ico_radius)
    phi = np.arctan2(ico_coords[:, 1], ico_coords[:, 0])

    #get Spherical Harmonics expansion coefficients (up to degree sph_degree)
    m = list(itertools.chain.from_iterable([[i for i in range(-n,n+1)] for n in range(sph_degree)]))
    m = np.array(m)

    l = list(itertools.chain.from_iterable([[k for _ in range(2 * k + 1)] for k in range(sph_degree)]))
    l = np.array(l)

    Y = sph_harm(m[None, :], l[None, :], theta[:, None], phi[:, None])

    #sample sphere neighbors for each voxel in patch and interpolate intensity
    mapped_coords = patch_coords[None, :, :] + ico_coords[:, None, :]
    mapped_int = map_coordinates(patch, mapped_coords.T, order = 3)
    center_int = patch[ico_radius : -ico_radius, ico_radius : -ico_radius, ico_radius : -ico_radius]

    #Compute kurtosis (for better rotation invariance)
    kurt = kurtosis(mapped_int)

    #Apply sign function and pass obtain spherical expansion coefficients for each sample
    f = np.greater_equal(center_int.ravel()[:, None], mapped_int).astype('int')
    c = f.dot(Y)

    #obtain frequency components of threshold function by integrating and normalizing over orders m
    f = np.multiply(c[:, None, l == 0], Y[None, :, l == 0])
    for n in range(1, sph_degree):
        f = np.concatenate((f, np.sum(np.multiply(c[:, None, l == n], Y[None, :, l == n]),
                            axis=2, keepdims=True)), axis=2)
    f = np.sqrt(np.sum(f**2, axis=1))

    #keep real parts of decomposition and kurtosis
    f = np.real(f)
    kurt = np.real(kurt)

    #extract histograms
    H = np.histogram(kurt, bins = n_bins)[0]
    for i in range(sph_degree):
        H = np.column_stack((H, np.histogram(f[:, i], bins = n_bins)[0]))
    H = normalize(H, axis = 0)

    #Return Descriptor (concatenated or aggregated histograms)
    if concatenate is True:
        D = H.T.ravel()
    else: 
        D = H.sum(axis = 1)
    D = D / np.linalg.norm(D)

    return D



# -------------------------------------------------------------------- #
# 3D - LBP Implementation : Rotational Invariance & Pattern Uniformity #
# -------------------------------------------------------------------- #
def lbp_ri_u(point, img, patch_size, ico_radius, ico_level, n_bins = 20, joint = True):
    '''
    Computes 3D rotation-invariant uniform LBP descriptor for a patch centered around a given voxel.
    Each voxel in the patch assumes an LBP code via a region-growing algorithm operating on a neighbrohood graph 
    of a set of points sampled from an icosahedron.
    Various functions for producing codes are considered:
        - The signed intensity difference between neighboring & center voxels
        - The signed intensity difference between neighboring voxels and mean neighborhood intensity
        - The signed radial intensity difference between neighboring voxels residing in homocentric spheres 
          of different radius

    References: 1. Comparison between 2D and 3D Local Binary Pattern Methods for Characterization of Three-Dimensional Textures,
                   Paulhac L., Makris P., Rame Y.J., ICIAR (2008), LNCS 5112, pp. 670-679, 2008 
                2. Extended three-dimensional rotation invariant local binary patters,
                   Citraro L., Mahmoodi S., Darekar A., Vollmer B. Image and Vision Computing, 62, 
                   pp. 8-18, 2017, Elsevier, doi: http://dx.doi.org/10.1016/j.imavis.2017.03.004
                3. Extended local binary patterns for texture classification,
                   Liu L., Zhao L., Long Y., Kuang G., Fiegth P., Image and Vision Computing, 30,
                   pp. 86-99, 2012, Elsevier, doi: 10.1016/j.imavis.2012.01.001

    ...

    Parameters
    ----------
    point : tuple
        the point to be described
    img : ndarray (width x height x depth)
        the MR image
    patch_size : int
        size of cellular patch around point
    ico_radius : float
        radius of icosahedron to uniformly sample intensities around patch voxels
    ico_level : int
        controls refinement level of icosahedron
    n_bins : int
        number of bins for LBP histograms
    joint : bool
        if true, compute joint histogram of LBP variants, otherwise concatenate

    Returns
    ------- 
    D : ndarray 
        LBP descriptor

    '''

    #extract image patch
    psize = patch_size // 2
    x, y, z = point
    patch = img[x - psize - int(math.ceil(ico_radius)) : x + psize + int(math.ceil(ico_radius)) + 1,
                y - psize - int(math.ceil(ico_radius)) : y + psize + int(math.ceil(ico_radius)) + 1,
                z - psize - int(math.ceil(ico_radius)) : z + psize + int(math.ceil(ico_radius)) + 1]
    
    patch_coords = cartesian((range(patch_size), range(patch_size), range(patch_size))) + 1    

    #construct icosahedron for uniform sampling on sphere surface
    ico = icosphere(subdivisions = ico_level, radius = ico_radius)
    ico_coords = np.array(ico.vertices)
    theta = np.arccos(ico_coords[:, 2] / ico_radius)
    phi = np.arctan2(ico_coords[:, 1], ico_coords[:, 0])

    #sample sphere neighbors for each voxel in patch and interpolate intensity
    mapped_coords = patch_coords[None, :, :] + ico_coords[:, None, :]
    mapped_int = map_coordinates(patch, mapped_coords.T, order = 3)
    
    #different variant of LBP
    center_int = patch[ico_radius : -ico_radius, ico_radius : -ico_radius, ico_radius : -ico_radius]
    mean_int = mapped_int.mean(axis = 1)

    thres_center = np.greater_equal(center_int.ravel()[:, None], mapped_int).astype('uint8')
    thres_mean = np.greater_equal(mean_int[:, None], mapped_int).astype('uint8')

    #construct graphs for each mode of LBP codes
    edges = ico.edges_unique

    #1. Center voxel intensity thresholding
    Graphs_center = [nx.Graph() for _ in range(patch_size**3)]
    n_cc_center = []
    for g_idx, graph in enumerate(Graphs_center):
        node_labels = thres_center[g_idx, :]
        [graph.add_path(edge) for edge in edges if node_labels[edge[0]] == node_labels[edge[1]]]
        n_cc_center.append(len(list(nx.connected_components(graph))))
    n_cc_center = np.array(n_cc_center)

    #2. Mean spherical neighborhood intensity thresholding
    Graphs_mean = [nx.Graph() for _ in range(patch_size**3)]
    n_cc_mean = []
    for g_idx, graph in enumerate(Graphs_mean):
        node_labels = thres_mean[g_idx, :]
        [graph.add_path(edge) for edge in edges if node_labels[edge[0]] == node_labels[edge[1]]]
        n_cc_mean.append(len(list(nx.connected_components(graph))))
    n_cc_mean = np.array(n_cc_mean)

    #extract LBP codes for every voxel in patch for all methods
    u_idx_center = np.where(n_cc_center <= 3)[0]
    codes_center = ico.vertices.shape[0] * np.ones(shape = (patch_size**3, ))
    codes_center[u_idx_center] = thres_center[u_idx_center, :].sum(axis = 1)

    u_idx_mean = np.where(n_cc_mean <= 3)[0]
    codes_mean = ico.vertices.shape[0] * np.ones(shape = (patch_size**3, ))
    codes_mean[u_idx_mean] = thres_mean[u_idx_mean, :].sum(axis = 1)

    #Compute histogram (joint vs concatenated)
    if joint is True:
        hist = np.histogram2d(x = codes_center, y = codes_mean, bins = n_bins)[0]
        hist = hist.ravel()
    else:
        hist_center = np.histogram(codes_center, bins = n_bins)[0]
        hist_mean = np.histogram(codes_mean, bins= n_bins)[0]
        hist = np.hstack((hist_center, hist_mean))

    #return descriptor
    D = hist / np.linalg.norm(hist)
    return D
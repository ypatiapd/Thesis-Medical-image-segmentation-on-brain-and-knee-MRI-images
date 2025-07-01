"""
-------------------------------------------------------------------
3D - HOG feature extraction

1. Fully 3D HOG extension

2. Triplanar (HOG features in three orthogonal planes [xy, yz, xz])
-------------------------------------------------------------------
"""

import math

import numpy as np

# icosahedron building
from stripy import sTriangulation
from trimesh.creation import icosphere

from skimage.feature import hog

from sklearn.utils.extmath import cartesian
from sklearn.metrics.pairwise import linear_kernel

from Encoding.utils import getIcosahedron
from Encoding.utils import cartesianToSpherical
from Encoding.utils import getGradients
from Encoding.utils import getAverageMask
from Encoding.utils import structure_tensor_3D




# --------------------------------------------- #
# Tri-planar (-xy, -yz, -xz) 2D-HOG descriptors #
# --------------------------------------------- #

def triplanar_hog(point, image, patch_size, n_bins, n_cells = 1, normalize = True):
    """
    Naive extension o Histogram of Oriented Gradients in 3D
    Calculating HOG descriptors for each plane intersecting the voxel (xy-, yz-, xz-)
    Final descriptor as a result of concatenation of planar descriptors.

    ...

    Parameters
    ----------
    point : tuple
        voxel cartesian coordinates
    image : memmap
        MRI containing voxel
    patch_size : int
        size of cubic patch around point for HOG extraction
    n_bins : int
        number of orientation histogram bins
    n_cells : int
        number of disjoint cubic cells to compartmentalize patch
    normalize : bool
        whether to normalize histogram norm or not

    Returns
    -------
    d : ndarray
        voxel HOG descriptor

    """

    x, y, z = point

    img_xy = image[:, :, z]
    img_yz = image[x, :, :]
    img_xz = image[:, y, :]

    step = patch_size // 2
    patch_xy = img_xy[x - step - 1 : x + step + 2,
                      y - step - 1 : y + step + 2]

    patch_yz = img_yz[y - step - 1 : y + step + 2,
                      z - step - 1 : z + step + 2]

    patch_xz = img_xz[x - step - 1 : x + step + 2,
                      z - step - 1 : z + step + 2]

    cell_size = patch_size // n_cells

    #encode each patch
    d_xy = hog(image = patch_xy, orientations = n_bins, pixels_per_cell = (cell_size, cell_size),
               cells_per_block = (n_cells, n_cells), feature_vector = True, multichannel = False)
    d_xz = hog(image = patch_xz, orientations = n_bins, pixels_per_cell = (cell_size, cell_size),
               cells_per_block = (n_cells, n_cells), feature_vector = True, multichannel = False)
    d_yz = hog(image = patch_yz, orientations = n_bins, pixels_per_cell = (cell_size, cell_size),
               cells_per_block = (n_cells, n_cells), feature_vector = True, multichannel = False)

    d = np.hstack((d_xy, d_yz, d_xz))

    if normalize:
        d = d / np.linalg.norm(d)

    return d



def hog_3d_interp(point, image, patch_size, ico_level = 1, mode = 'aggregate', rot_inv = False, norm = 'l2'):
    """
    Extracts HOG-based feature descriptors from 3D MR images on a per-voxel base.
    Orientation binning is implemented by assigning gradient orientations
    to the faces of a regular icosahedron, interpolating votes to encompassing vertices.
    Optional rotation invariance through the structure tensor.

    ...

    Parameters
    ----------
    point : array_like (3, )
        voxel cartesian coordinates
    image : numpy memmap (N, N, M)
        MRI containing voxel
    rsize : int
        size of region around central voxel
    psize : int
        size of patches in region
    ico_level : int
        number of times to subdivide icosahedron for finer description
    mode : str
        variable indicating whether histograms of constituent patches will be aggregated or concatenated
    rot_inv : bool
        boolean variable indicating whether to rotate image patch according to dominant orientation
        for rotational invariance
    norm : str, None
        variable indicating method (or lack of) descriptor normalization


    Returns
    -------
    D : ndarray (q, )
        voxel descriptor

    """


    # Create icosahedron
    ico = getIcosahedron(ico_level, ico_type = 'stripy')

    x, y, z = point

    step = patch_size // 2
    image_patch = image[x - step - 1 : x + step + 2,
                        y - step - 1 : y + step + 2, 
                        z - step - 1 : z + step + 2]
    magnitude, phi, theta = getGradients(image_patch, coords = 'sph')

    gmask = getAverageMask(patch_size)

    magnitude = (gmask * magnitude).ravel()
    latitudes = theta.ravel()
    longitudes = phi.ravel()

    histogram = np.zeros(shape = (ico.npoints, ))
    barycentric_coords, vertices = ico.containing_simplex_and_bcc(lats = latitudes, lons = longitudes)
    for point_idx, (bcc, vertex) in enumerate(zip(barycentric_coords, vertices)):
        histogram[vertex] += magnitude[point_idx] * bcc
            
    if norm == 'l2':
        histogram /= np.linalg.norm(histogram)
    elif histogram == 'l2-hys':
        histogram = np.clip(histogram, 0, 0.2)
        histogram /= np.linalg.norm(histogram)
    else:
        pass
                
    return histogram



def hog_3d_proj(point, image, psize = 5, rsize = 15, orientation = 'vertices', level = 1, mode = 'aggregate', 
                rot_inv = False, norm = 'l2'):
    '''
    Computes a 3D variant of the HOG Descriptor for an image region centered arounda voxel
    The Region of size (rsize x rsize x rsize) is compartmentalized into a set of disjoint patches,
    each of size (psize x psize x psize). A histogram of oriented gradients is computed for each patch, 
    with the orientation bins corresponding to vertices of centers of faces of a regular (refined) icosahedron.
    The final descriptor is a weighted aggregate of those histograms. Currently, implementation supports regions arranged in
    3x3x3 patches.

    Reference: Alexander Klaser, Marcin Marszalek, Cordelia Schmid. 
               A Spatio-Temporal Descriptor Based on 3D-Gradients. 
               BMVC 2008 - 19th British Machine Vision Conference, Sep 2008, Leeds, United Kingdom.pp.275:1-10. 
               DOI:10.5244/C.22.99

    ...

    Parameters
    ----------
    point : array - like
        the voxels to be characterized
    image : ndarray
        the image containing the voxels
    psize : int
        size of patches in region
    rsize : int
        size of region around central voxel
    orientation : string
        whether to associate histogram bins with vertices of centroids of faces of the icosahedron
    ico_coords : string
        coordinate system to define icosahedron on
    level : int
        number of refienement steps for icosahedron
    mode : string
        chooses whether to concatenate or aggregate patch histograms to form final descriptor
    
    Returns
    -------
    D : ndarray
        voxel descriptor

    '''

    #sanity check
    assert type(rsize // psize) == int, print("Wrong combination of regional and patch sizes")

    #set params
    rs = rsize // 2
    ps = psize // 2
    ncells = rsize // psize
    
    # get icosahedron
    ico = icosphere(subdivisions = level)
    if orientation  == 'faces':
        axes = np.array(ico.face_normals)
    else:
        axes = np.array(ico.vertices)

    # get average masks
    region_mask = getAverageMask(rsize // psize, 'manhattan')
    patch_mask = getAverageMask(psize, 'manhattan')

    #calculate partial derivatives
    x, y, z = point 
    xp = range(- rs + ps, rs - ps + 1, psize)
    yp = range(- rs + ps, rs - ps + 1, psize)
    zp = range(- rs + ps, rs - ps + 1, psize)
    patch_centers = cartesian((xp, yp, zp))
    patch_locations = patch_centers + psize

    # extracting +1 voxel in each direction for computational consistency
    region = image[x - rs - 1 : x + rs + 2,
                   y - rs - 1 : y + rs + 2,
                   z - rs - 1 : z + rs + 2]
        
    i_dx, i_dy, i_dz = getGradients(region)

    #get gradients at the patch level
    dx = np.array([i_dx[ploc[0] : ploc[0] + psize, ploc[1] : ploc[1] + psize,
                        ploc[2] : ploc[2] + psize] for ploc in patch_locations])

    dy = np.array([i_dy[ploc[0] : ploc[0] + psize, ploc[1] : ploc[1] + psize,
                        ploc[2] : ploc[2] + psize] for ploc in patch_locations])
    
    dz = np.array([i_dz[ploc[0] : ploc[0] + psize, ploc[1] : ploc[1] + psize,
                        ploc[2] : ploc[2] + psize] for ploc in patch_locations])

    dx = dx.reshape((ncells**3, psize**3))
    dy = dy.reshape((ncells**3, psize**3))
    dz = dz.reshape((ncells**3, psize**3))

    #collect all gradients in one array and calculate magnitudes
    raw_gradients = np.dstack((dx, dy, dz))
    if rot_inv is True:
        #rotate region according to dominant direction to achieve rotational invariance
        R = structure_tensor_3D(raw_gradients, getAverageMask(rsize, 'gaussian'))
        gradients = R.T.dot(raw_gradients.reshape((-1, 3)).T) 
        gradients = gradients.reshape(3, raw_gradients.shape[1], raw_gradients.shape[0]).T
    else:
        gradients = raw_gradients
    gradient_magnitudes = np.linalg.norm(gradients, axis = 2)

    #project gradients to icosahedron orientation axes
    projected_gradients = gradients.dot(axes.T)
    projected_gradients /= gradient_magnitudes[:, :, None]

    # compute theshold to clip projected gradients and recalculate magnitude
    inner_prods = linear_kernel(axes)[0, :]
    thres = np.sort(inner_prods)[-2]

    projected_gradients -= thres
    projected_gradients[projected_gradients < 0] = 0
    projected_gradient_magnitudes = np.linalg.norm(projected_gradients, axis = 2)

    #distribute original magnitude in orientation bins
    gradient_histograms = projected_gradients * (gradient_magnitudes[:, :, None] / projected_gradient_magnitudes[:, :, None])
    D = gradient_histograms.sum(axis = 1)

    if mode == 'flatten':
        Descriptor = (region_mask.ravel()[:, None] * D).ravel()
    else:
        Descriptor = region_mask.ravel().dot(D)

    if norm == 'l2':
        Descriptor = Descriptor / np.linalg.norm(Descriptor)
    if norm == 'l2-hys':
        Descriptor = Descriptor / np.linalg.norm(Descriptor)
        Descriptor = np.clip(Descriptor, a_min = 0, a_max = 0.25)
        Descriptor = Descriptor / np.linalg.norm(Descriptor)

    return Descriptor



def hog_3d_sph(point, image):
    pass
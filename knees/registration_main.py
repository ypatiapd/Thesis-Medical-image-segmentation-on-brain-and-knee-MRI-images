'''
Multi-Atlas-Registration main script
'''

import os
import sys
import gc
import shutil

import numpy as np
import SimpleITK as sitk

import pandas as pd

from functools import partial
from multiprocessing import Pool, cpu_count

from Registration.registration import register
from Registration.registration import dilate_average_mask
from Preprocessing.ImagePreprocessing import rescale

from Registration.utils import sitk_to_numpy
from Registration.utils import remove_empty_dirs
from Registration.utils import remove_bad_atlases
from Registration.utils import get_dice_scores


#Source dir contains initial MRIs & Masks
#Output dir contains registered images and segmentation results
source_dir = sys.argv[1]
output_dir = sys.argv[2]


subjects = os.listdir(source_dir)
subjects.sort()

#Number of availabe CPU cores
#n_cores = cpu_count()
n_cores = 3

#Each separate subject is a target with the rest as atlases
for subject in subjects:
    
    # ---------------------- #
    # Subject identification #
    # ---------------------- #
    target = subject
    atlases = subjects.copy()
    atlases.remove(subject)
    atlases.sort()


    # ---------------- #
    # Directory set-up #
    # ---------------- #
    target_output_dir = os.path.join(output_dir, target)
    if not os.path.isdir(target_output_dir):
        os.mkdir(target_output_dir)

    target_input_dir = os.path.join(source_dir, target)
    atlas_input_dirs = [os.path.join(source_dir, atlas) for atlas in atlases]


    # ------------------- #
    # Save target subject #
    # ------------------- #
    target_mri = sitk_to_numpy(sitk.ReadImage(os.path.join(target_input_dir, 'mri.mhd')))
    target_mask = sitk_to_numpy(sitk.ReadImage(os.path.join(target_input_dir, 'mask.mhd')))
    np.save(os.path.join(target_output_dir, 'mri.npy'), target_mri)
    np.save(os.path.join(target_output_dir, 'mask.npy'), target_mask)


    # ------------------------ #
    # Multi-atlas registration #
    # ------------------------ #
    pool = Pool(n_cores)
    SSD = pool.map(partial(register, target_dir = target_input_dir, output_dir = output_dir), atlas_input_dirs)
    SSD = np.asarray(SSD)
    np.save(os.path.join(target_output_dir, 'SSD.npy'), SSD)


    # --------------------------- #
    # Remove failed registrations #
    # --------------------------- #
    atlases = remove_empty_dirs(target_output_dir, atlases, 4)
    atlases.sort()


    # ------------------------- #
    # Selection of best atlases #
    # ------------------------- #
    dice = get_dice_scores(target_output_dir)
    dice = np.asarray(dice)
    indices = np.argsort(dice)
    best = indices[-20:]
    worst = np.delete(indices, best)
    selected_atlases = [atlases[i] for i in best]
    discarded_atlases = [atlases[i] for i in worst]
    remove_bad_atlases(target_output_dir, discarded_atlases)


    # -------------------------------------------------------- #
    # Save average & dilated mask for 5 - 10 - 15 - 20 atlases #
    # -------------------------------------------------------- #
    for atlas_num in [5, 10, 15, 20]:

        dirname = str(atlas_num) + '_Atlases'
        if not os.path.isdir(os.path.join(target_output_dir, dirname)):
            os.mkdir(os.path.join(target_output_dir, dirname))

        selected_atlas_dirs = [os.path.join(target_output_dir, atlas) for atlas in selected_atlases[-atlas_num:]]
        average_mask, average_dilated_mask = dilate_average_mask(selected_atlas_dirs)
    
        average_mask[average_mask == 3] = 0
        average_dilated_mask[:, :, :15] = 0
        average_dilated_mask[:, :, -15:] = 0

        #save
        corr_atlases = np.asarray(selected_atlases[-atlas_num:])
        np.save(os.path.join(target_output_dir, dirname, 'atlases.npy'), corr_atlases)
        np.save(os.path.join(target_output_dir, dirname, 'average_mask.npy'), average_mask)
        np.save(os.path.join(target_output_dir, dirname, 'sampling_area.npy'), average_dilated_mask)
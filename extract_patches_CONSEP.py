"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import shutil

import skimage.io
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
import warnings
warnings.filterwarnings("ignore")
from dataset import get_dataset
import skimage.io as io
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"
    xtractor = PatchExtractor(win_size, step_size)
    # monudir='/data1/wyj/M/datasets/MoNuSACGT/stage1_train/'#need to change!!!!!!!!!!!!
    monudir = '/data1/wyj/M/datasets/consepGT/Train/'  # need to change!!!!!!!!!!!!
    pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm.tqdm(
        total=len(os.listdir(monudir)), bar_format=pbar_format, ascii=True, position=0
    )
    hvdir = '/data1/wyj/M/datasets/consepGT/trainHV/'
    if os.path.exists(hvdir):
        shutil.rmtree(hvdir)
    if not os.path.exists(hvdir):
        os.mkdir(hvdir)
    for dirname in os.listdir(monudir):
        imgpath=monudir+dirname+'/images/'+dirname+'.png'
        maskspath = monudir + dirname + '/masks/'
        img = io.imread(imgpath)
        img_mask=np.zeros((1000,1000,1),dtype=np.int)
        img_mask_type = np.zeros((1000, 1000, 1), dtype=np.int)
        count=0
        ANNO_sum=0
        for maskp in os.listdir(maskspath):
            count+=1
            tempmask=io.imread(os.path.join(maskspath,maskp))
            ANNO_sum+= np.sum(tempmask!=0)
            img_mask[tempmask!=0,0]=count
            img_mask_type[tempmask != 0, 0] = 1
        print('ANNO_SUM:{}   FINAL_SUM:{}'.format(ANNO_sum,np.sum(img_mask!=0)))

        img = np.concatenate([img, img_mask,img_mask_type], axis=2)
        sub_patches = xtractor.extract(img, extract_type)

        pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbar = tqdm.tqdm(
            total=len(sub_patches),
            leave=False,
            bar_format=pbar_format,
            ascii=True,
            position=1,
        )
        VAL_IMAGE_IDS = [
            "TCGA-E2-A1B5-01Z-00-DX1",
            "TCGA-E2-A14V-01Z-00-DX1",
            "TCGA-21-5784-01Z-00-DX1",
            "TCGA-21-5786-01Z-00-DX1",
            "TCGA-B0-5698-01Z-00-DX1",
            "TCGA-B0-5710-01Z-00-DX1",
            "TCGA-CH-5767-01Z-00-DX1",
            "TCGA-G9-6362-01Z-00-DX1",

            "TCGA-DK-A2I6-01A-01-TS1",
            "TCGA-G2-A2EK-01A-02-TSB",
            "TCGA-AY-A8YK-01A-01-TS1",
            "TCGA-NH-A8F7-01A-01-TS1",
            "TCGA-KB-A93J-01A-01-TS1",
            "TCGA-RD-A8N9-01A-01-TS1",
        ]
        if not os.path.exists('{}masks-newori'.format(hvdir)):
            os.mkdir('{}masks-newori'.format(hvdir))
        if not os.path.exists('{}masks-instance'.format(hvdir)):
            os.mkdir('{}masks-instance'.format(hvdir))
        if not os.path.exists('{}masks-new'.format(hvdir)):
            os.mkdir('{}masks-new'.format(hvdir))
        if not os.path.exists('{}masks-new-valid'.format(hvdir)):
            os.mkdir('{}masks-new-valid'.format(hvdir))
        for idx, patch in enumerate(sub_patches):
            # skimage.io.imsave("{0}/{1}_{2:03d}.tif".format('/data1/wyj/M/datasets/MoNuSAC/masks-newori', dirname, idx), patch)
            skimage.io.imsave("{0}/{1}_{2:03d}.png".format('{}masks-newori'.format(hvdir), dirname, idx),
                              patch[:,:,0:3])
            skimage.io.imsave("{0}/{1}_{2:03d}.png".format('{}masks-instance'.format(hvdir), dirname, idx),
                              patch[:,:,3])
            if dirname[:23] not in VAL_IMAGE_IDS:
                np.save("{0}/{1}_{2:03d}.npy".format('{}masks-new'.format(hvdir), dirname, idx), patch)
            else:
                np.save("{0}/{1}_{2:03d}.npy".format('{}masks-new-valid'.format(hvdir), dirname, idx), patch)
            pbar.update()

        pbar.close()
        pbarx.update()
    pbarx.close()

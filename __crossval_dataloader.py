import torch
from copy import deepcopy
import numpy as np
import pathlib as pl
import pandas as pd


import bioformats
import javabridge as jb
import tifffile
import zarr
import os


import matplotlib

from matplotlib_scalebar.scalebar import ScaleBar
from resize.pytorch import resize
import albumentations as A




# def flip_aug(a,b,c ):
#     """Randomly flips along x-axis, y-axis, or both, or neither."""
#     if np.random.random() < 0.5:
#         a = torch.flip(a, dims=(1,))
#         b = torch.flip(b, dims=(1,))
#         c = torch.flip(c, dims=(1,))


#     if np.random.random() < 0.5:
#         a = torch.flip(a, dims=(2,))
#         b = torch.flip(b, dims=(2,))
#         c = torch.flip(c, dims=(2,))

#     return a, b, c

# def rot_aug(a,b,c):
#     """Randomly rotates in multiples of 90 degrees"""
#     k = np.random.choice([0, 1, 2, 3])

#     a = torch.rot90(a, k=k, dims=(1,2))
#     b = torch.rot90(b, k=k, dims=(1,2))
#     c = torch.rot90(c, k=k, dims=(1,2))

#     return a, b, c


def flip_aug(a ):
    """Randomly flips along x-axis, y-axis, or both, or neither."""
    if np.random.random() < 0.5:
        a = torch.flip(a, dims=(1,))


    if np.random.random() < 0.5:
        a = torch.flip(a, dims=(2,))


    return a

def rot_aug(a):
    """Randomly rotates in multiples of 90 degrees"""
    k = np.random.choice([0, 1, 2, 3])

    a = torch.rot90(a, k=k, dims=(1,2))


    return a

def inform(a):
    print(a.shape, a.dtype, type(a))



def clip(a, _min, _max):

    a[a < _min] = _min
    a[a > _max] = _max



class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, n_items, 
                 mapping_level_2, reverse_mapping_level_2, 
                 mapping_level_1, reverse_mapping_level_1, 
                 mapping_level_0, reverse_mapping_level_0, 
                 patch_size, training):
        
        self.dataframe = deepcopy( dataframe )
        self.n_items = n_items
        

        self.mapping_level_2 = mapping_level_2
        self.reverse_mapping_level_2 = reverse_mapping_level_2



        self.patch_size = patch_size
        self.training = training


    def __len__(self):
        return self.n_items


    def __getitem__(self, _index):

        if self.training:

            # Undersampling - even probability of looking at a patch with any of the labels
            # Uniformly pull a cell label, this is a string of the cell name
            cur_class = np.random.choice( list( self.mapping_level_2.values() )  )


            # Randomly grab a row in the dataframe with that label
            index = np.random.choice( self.dataframe[self.dataframe["level_2_label_string"] == cur_class].index.tolist() )


        else:
            index = _index


        # select relevant df row
        # datapoint = deepcopy( self.dataframe.loc[ index ] )
        datapoint = self.dataframe.loc[ index ] 



        # get new patch from centroid
        centroid_row = int( datapoint["centroid_row"] )
        centroid_col = int( datapoint["centroid_col"] )


        start_row = centroid_row - (self.patch_size // 2)
        start_col = centroid_col - (self.patch_size // 2)


        #=========================
        # Read vH&E

        # correct a missing intial slash
        vhe_path = "/" + datapoint["vhe"]

        if os.path.exists(vhe_path):
            store = tifffile.imread(vhe_path, aszarr=True)

        else:
            print("path doesn't exist:", vhe_path)
            return
        z = zarr.open(store, mode='r')
        start_row = centroid_row - (self.patch_size // 2)
        start_col = centroid_col - (self.patch_size // 2)

        X = z[start_row:start_row+self.patch_size, start_col:start_col+self.patch_size].astype(np.float32)
        X = np.moveaxis(X, -1, 0) # put channels first
        #=========================



        # normalize x
        X /= 255.0

        
        X = torch.tensor(X)
        # Y = torch.tensor(Y)
        # mask = torch.tensor(mask)


        if self.training:
            # X, Y, mask = flip_aug( X, Y, mask )
            # X, Y, mask = rot_aug( X, Y, mask )
            X = flip_aug( X )
            X = rot_aug( X )


        label_string_level_2 = datapoint["level_2_label_string"]
        label_level_2 = self.reverse_mapping_level_2[label_string_level_2]
        

        
        X = resize( X.unsqueeze(0), dxyz=(1.56, 1.56), order=3 )[0] # order 3 in cubic interp
        clip(X, 0, 1)


        if self.training:

            #==============================================================
            # realistic H&E color augmentation based on observation
            transform = A.Compose([
                A.HueSaturationValue(    
                    hue_shift_limit=0.4,
                    sat_shift_limit=0.4,
                    val_shift_limit=0.2,
                    always_apply=False,
                    p=0.5,),


            ])


            # move channels to the back before color augmentation. it must be numpy for this library's augmentation
            X = torch.moveaxis( X, 0, -1 ).numpy()

            # color augment

            transformed = transform( image = X )
            transformed_image = transformed["image"]

            # put it back in the batch
            X = transformed_image

            # convert back to torch tensor and move channels back 
            X = torch.tensor(X).float()
            X = torch.moveaxis(X, -1, 0)
            #==============================================================


        return X, label_level_2, torch.tensor([0]), torch.tensor([0])











import torch.nn as nn
import torch

import torchvision.models as models
import torchvision
import torch.utils.data 

from copy import deepcopy

import pandas as pd

import numpy as np

# import unet_model

import numpy as np
from datetime import date
import torch
import os

import cv2

import matplotlib.pyplot as plt

import sys
import pathlib as pl

from sklearn.model_selection import train_test_split

import monai
import pickle
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import torchvision
#sys.path.insert(0, "/home/remedilw/data/gca_he/MIDL_github")
import random
# import unet2D_ns_smaller

# import MOTSDataset_2D_Patch_normal
from PIL import Image
# import loss_functions.loss_2D as loss

# from train_2D_patch import get_loss
from copy import deepcopy
import random
import math

import gc
#
import pandas as pd

import torch
torch.backends.cudnn.benchmark = True # makes code faster through optimization of code

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import albumentations as A

# from ___dataloader_for_nucleus_patch_classification import Dataset

# from __074_dataloader import Dataset as nonaug_Dataset

from __crossval_dataloader import Dataset




example_training_df_path = "/nfs/masi/remedilw/paper_journal_nucleus_subclassification/nucleus_subclassification/training_data/fold_0_train_nuclei_05272023_written_with_vhe_163.csv"
example_val_df_path = "/nfs/masi/remedilw/paper_journal_nucleus_subclassification/nucleus_subclassification/training_data/fold_0_val_nuclei_05272023_written_with_vhe_163.csv"

# sys.path.insert(0, "/home/remedilw/code/gca_he/segmentation/qa_mxif")
# from __smaller_resnet import *

import datetime

import argparse

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--fold')


    return p



parser = _build_arg_parser()
args = parser.parse_args()

fold = args.fold




print("fold", fold, type(fold))



train_df = pd.read_csv(example_training_df_path)
val_df = pd.read_csv(example_val_df_path)


print("reading from these files")



print("n train examples", train_df.index)
print("n val examples", val_df.index)




    






mapping_level_0 = {
    0 : "Leukocyte",
    1 : "Epithelial",
    2 : "Connective",
}

mapping_level_1 = {

    0 : "Leukocyte",

    1 : "Lymphocyte",
    2 : "Myeloid",

    3 : "Epithelial",
    4 : "Goblet",

    5 : "Fibroblasts",

    6 : "Stromal"


}

# 13 cells. We dropped leukocyte because it didn't make sense in the tree
mapping_level_2 = {

    0 : "Leukocyte",

    1: 'B',
    2: 'T_Receptor',
    3: 'T_Helper',
    4: 'Cytotoxic_T',


    5: 'Myeloid',
    6: 'Monocytes',
    7: 'Macrophages',

    8: 'Epithelial',
    9: 'Progenitor',
    10: 'Endocrine',
    11: 'Goblet',

    12: 'Fibroblasts',
    13: 'Stromal',
}


reverse_mapping_level_0 = {v:k for k,v in mapping_level_0.items()}
reverse_mapping_level_1 = {v:k for k,v in mapping_level_1.items()}
reverse_mapping_level_2 = {v:k for k,v in mapping_level_2.items()}






now = datetime.datetime.now()

current_timestamp = "-".join([ str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second) ])

current_timestamp

#------------------------a--------------------------------------------------------
# Hyperparams
#--------------------------------------------------------------------------------
my_experiment_name = "__res_fold_{}_".format(fold)

my_experiment_name += current_timestamp



# THIS WILL BE DOWNSAMPLED TO 41X41 IN THE DATALOADER
patch_size = 64

lr = 1e-3
batch_size = 256

loss_obj = torch.nn.CrossEntropyLoss()

device_idx = 0


input_channels = 3
output_classes = 14






# disk_images = pl.Path("/home/remedilw/data/gca/debug_mxif_segmentation/mxif/stain_normalized_by_thresholds_float32")

# inst_masks_on_disk = pl.Path("/home/remedilw/data/gca/debug_mxif_segmentation/mxif/deepcell_label")

#==========================
# Seed
random.seed(0)
rand_state = 0
torch.manual_seed(0)
print()
#==========================

#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Check on cuda
#--------------------------------------------------------------------------------
print( "Is cuda available?:", torch.cuda.is_available() )

print( "How many gpus are available?", torch.cuda.device_count() )

#print( torch.cuda.current_device() )

#print( torch.cuda.device(     ) )

print( "Name of my specificed GPU:", torch.cuda.get_device_name(device_idx) )
#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------
# Setup for experiment
#--------------------------------------------------------------------------------
pwd = pl.Path().cwd()
print("PWD", pwd)

device = torch.device("cuda:{}".format(device_idx))

writer = SummaryWriter()

today = date.today()
day_and_time = today.strftime("%b-%d-%Y")

#==========================
#==========================
#==========================
#==========================
experiment_name = "experiments/{}_{}".format(my_experiment_name, day_and_time)
#==========================
#==========================
#==========================
#==========================


outdir = pl.Path(experiment_name)
if not os.path.exists(outdir):
    os.makedirs(outdir)


tb_outdir = pl.Path("runs/TB_{}".format(experiment_name))
if not os.path.exists(tb_outdir):
    os.makedirs(tb_outdir)
#--------------------------------------------------------------------------------





#-------------a-------------------------------------------------------------------
#### Find the subset of data for the current fold's train and val
#--------------------------------------------------------------------------------



### Check that there is no onverlap between subjects or images in the the train and val data
# #--------------------------------------------------------------------------------
# no overlapping images
assert set( train_df.image.tolist() ).intersection( set(val_df.image.tolist())  ) == set()
assert set( val_df.image.tolist() ).intersection( set(train_df.image.tolist())  ) == set()



#--------------------------------------------------------------------------------
### How many steps do we train for?
# #--------------------------------------------------------------------------------
# Grab a balanced subset from the val set 
n_val_from_each_class = 100

selected_indices_for_balancing_val_set = []

# use level 2 mapping -- all 12 classes
for cell_str in mapping_level_2.values():

    subset = val_df[val_df["level_2_label_string"] == cell_str]

    rand_examples_from_subset = random.sample(  subset.index.tolist(), n_val_from_each_class )

    selected_indices_for_balancing_val_set += rand_examples_from_subset


val_df = val_df.loc[selected_indices_for_balancing_val_set].reset_index(drop=True)
print("{} val samples balanced across classes".format(len(val_df)))
###########




n_train_items = len(train_df) * 100
n_val_items = len(val_df)

#n_steps = math.ceil( n_train_items / batch_size )
n_steps = 20000



#--------------------------------------------------------------------------------
### Datasets
# #--------------------------------------------------------------------------------

train_ds = Dataset(dataframe=train_df, 
                    n_items=n_train_items, 
                    mapping_level_2=mapping_level_2, 
                    reverse_mapping_level_2=reverse_mapping_level_2,
                    mapping_level_1=mapping_level_1, 
                    reverse_mapping_level_1=reverse_mapping_level_1,
                    mapping_level_0=mapping_level_0, 
                    reverse_mapping_level_0=reverse_mapping_level_0,
                    patch_size=patch_size,
                    training=True)


val_ds = Dataset(dataframe=val_df, 
                 n_items=n_val_items, 
                 mapping_level_2=mapping_level_2, 
                 reverse_mapping_level_2=reverse_mapping_level_2,
                 mapping_level_1=mapping_level_1, 
                 reverse_mapping_level_1=reverse_mapping_level_1,
                 mapping_level_0=mapping_level_0, 
                 reverse_mapping_level_0=reverse_mapping_level_0,
                 patch_size=patch_size,
                 training=False)


train_gen = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=False, # random sampling already
                    pin_memory=True,
                    num_workers=20,
                    prefetch_factor=3
            )

val_gen = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=32, # need batch of 1 for val
                    shuffle=False, # validation
                    pin_memory=True,
                    num_workers=8,
                    # prefetch_factor=5
            )




#--------------------------------------------------------------------------------
### Model
# #--------------------------------------------------------------------------------

model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")

# change input channels to match image dimensions
model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.bn1 = torch.nn.InstanceNorm2d(64, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer1[0].bn1 = torch.nn.InstanceNorm2d(64, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer1[0].bn2 = torch.nn.InstanceNorm2d(64, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer1[1].bn1 = torch.nn.InstanceNorm2d(64, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer1[1].bn2 = torch.nn.InstanceNorm2d(64, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)



model.layer2[0].bn1 = torch.nn.InstanceNorm2d(128, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer2[0].bn2 = torch.nn.InstanceNorm2d(128, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer2[0].downsample[1] = torch.nn.InstanceNorm2d(128, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)


model.layer2[1].bn1 = torch.nn.InstanceNorm2d(128, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer2[1].bn2 = torch.nn.InstanceNorm2d(128, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)




model.layer3[0].bn1 = torch.nn.InstanceNorm2d(256, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer3[0].bn2 = torch.nn.InstanceNorm2d(256, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer3[0].downsample[1] = torch.nn.InstanceNorm2d(256, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)


model.layer3[1].bn1 = torch.nn.InstanceNorm2d(256, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer3[1].bn2 = torch.nn.InstanceNorm2d(256, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)



model.layer4[0].bn1 = torch.nn.InstanceNorm2d(512, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer4[0].bn2 = torch.nn.InstanceNorm2d(512, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer4[0].downsample[1] = torch.nn.InstanceNorm2d(512, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)


model.layer4[1].bn1 = torch.nn.InstanceNorm2d(512, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)

model.layer4[1].bn2 = torch.nn.InstanceNorm2d(512, 
                                    eps=1e-05, 
                                    momentum=0.1,
                                    affine=False, 
                                    track_running_stats=False)



model.fc = torch.nn.Linear(in_features=512, out_features=output_classes, bias=True)




# model = ResNet18(input_channels, output_classes)

model = model.to(device)



#--------------------------------------------------------------------------------
### Optimizer
# #--------------------------------------------------------------------------------
opt = torch.optim.AdamW(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=lr,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )
opt.step()
#--------------------------------------------------------------------------------






#--------------------------------------------------------------------------------
### Training
# #--------------------------------------------------------------------------------

scaler = torch.cuda.amp.GradScaler()

# how often to perform validation
#val_freq = len(train_df) / batch_size``
val_freq = 200

loss_obj = loss_obj.to(device)

# nll = torch.nn.NLLLoss(reduction="none").to(device)



my_current_step = 0


with tqdm(total=n_steps) as pbar:


    pbar_dict = {
        "loss": np.finfo(np.float32).max,
        "val_loss": np.finfo(np.float32).max,

    }

    for cur_batch, (img_cpu, label_2_cpu, label_1_cpu, label_0_cpu ) in enumerate(train_gen):
        # print(my_current_step)

        model.train()

        with torch.cuda.amp.autocast():

            img = img_cpu.to(device)
            label_2 = label_2_cpu.to(device)
            # label_1 = label_1_cpu.to(device)
            # label_0 = label_0_cpu.to(device)


            opt.zero_grad()

            # logits
            pred = model(img)



            loss_2 = loss_obj(pred, label_2)
            loss = loss_2




            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        pbar_dict["loss"] = loss.detach().cpu().numpy().item()

        pbar.set_postfix(
                {
                    k: f"{v:.4f}"
                    for k, v in pbar_dict.items()
                    if v != 0 and v != np.finfo(np.float32).max
                }
            )
        pbar.update(1)
        scheduler.step()


        writer.add_scalars( str(tb_outdir / 'loss'), {
            'train': loss.detach().cpu().numpy().item(),

        }, cur_batch)

        gc.collect()


        if cur_batch % val_freq == 0:
            torch.save(model.state_dict(), outdir/"epoch_{}_weights.pth".format(cur_batch))
            # torch.save(opt.state_dict(), outdir/"epoch_{}_optimizer.pth".format(cur_batch))


            # run validation
            model.eval()

            with torch.no_grad():

                avg_val_loss = 0.0
                for cur_val_batch, (img_cpu, label_2_cpu, label_1_cpu, label_0_cpu ) in tqdm( enumerate(val_gen), total=len(val_gen) ):

                    with torch.cuda.amp.autocast():

                        img = img_cpu.to(device)
                        label_2 = label_2_cpu.to(device)
                        # label_1 = label_1_cpu.to(device)
                        # label_0 = label_0_cpu.to(device)

                        pred = model(img)


                        loss_2 = loss_obj(pred, label_2)
                       

                        loss = loss_2

                        val_loss = loss

                        avg_val_loss += val_loss

            # normlaize by the number of batches
            # we are using batch size 1. first batch has cu_val_batch == 0
            # n batches is cur_val_batch + 1 at the end
            avg_val_loss /= cur_val_batch+1

            writer.add_scalars( str(tb_outdir / 'loss'), {
                'val': avg_val_loss,

            }, cur_batch)
            pbar_dict["val_loss"] = avg_val_loss.detach().cpu().numpy().item()


            pbar.set_postfix(
                    {
                        k: f"{v:.4f}"
                        for k, v in pbar_dict.items()
                        if v != 0 and v != np.finfo(np.float32).max
                    }
                )

        my_current_step += 1

        if my_current_step >= n_steps - 1:

            writer.flush()

            print("at final step, leaving")
            exit()
    
##########################
##########################
##########################



##########################
##########################
##########################

# #--------------------------------------------------------------------------------













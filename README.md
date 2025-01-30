
# Nucleus Segmentation & Subclassification on H&E

<img width="654" alt="image" src="https://github.com/user-attachments/assets/32372096-37fd-48c0-81f8-91deb485d4e7" />


Publication: [Data-driven nucleus subclassification on colon hematoxylin and eosin using style-transferred digital pathology](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-11/issue-6/067501/Data-driven-nucleus-subclassification-on-colon-hematoxylin-and-eosin-using/10.1117/1.JMI.11.6.067501.full)


# This repo enables nucleus/cell classification on intestinal H&E into 14 classes

![Description of Image](./cell_classification.png)

This image shows a zoomed in section of a whole slide image of virtual H&E, where nuclei/cells have been classified into 14 classes.

# Note: 
**Pretrained weights are currently not public. They are only available to members of the MASI lab.**

# Brief Overview
This repo provides pretrained models (as well as code to train new models).
The pretrained models were trained on virtual H&E to classify nuclei into 14 classes.

**If you want to use this model on real H&E:**
It is recommended for the H&E staining to be style transferred to the virtual H&E style.
You can do that following instructions here: https://github.com/MASILab/he_stain_translation_cyclegan

Then the nuclei must be located. 
You can do that following instructions here: https://github.com/MASILab/hovernets_on_vhe



# Citations
If you use this repo, please cite
- "Data-driven Nucleus Subclassification on Colon H&E using Style-transferred Digital Pathology"

# Pretrained weights
The weights for 5 folds of trained resnets for nucleus classification on virtual H&E can be found here:

**MASI Lab:** ```/nfs/masi/remedilw/paper_journal_nucleus_subclassification/nucleus_subclassification/weights```

**Public:** Not currently available

# Inference
An example jupyter notebook showing how a pretrained model can be run on a whole slide image (virtual H&E) can be found in this repo
```inference_on_wsi.ipynb```

If you are in MASI lab, the paths are setup so that this inference notebook will run with the example data.

If you are not in MASI lab, change paths accordingly.

The pretrained models expect to perform inference on virtual H&E or H&E with resolution: 0.5 microns per pixel (mpp)

# Training
Training can be run using
```python __crossval_resnet_20k-steps_batch_256.py```

If you are in MASI lab, data necessary has been moved here:
```/nfs/masi/remedilw/paper_journal_nucleus_subclassification/nucleus_subclassification/training_data```

The csvs and associated data are in that folder, though you will need to make a copy of the csvs and update the parent paths to the images once you move the data locally to a machine.

If you are not in the MASI lab, then the training script can be examined, but basically it expects
1) Whole Slide Images of H&E or virtual H&E
2) A csv, where each row is a nucleus, with a centroid (row and column), and a class label (cell type)

The resnet approach is simple. Given the coordinates of a nucleus, the dataloader reads a small patch around the nucleus, from the whole slide image (on the fly).
Then it learns to predict the center nucleus in the patch, given the class label.

# Note
Please note that in the provided scripts, the training involved on-the-fly resampling of virtual H&E patches in the dataloader.
However, in the provided inference script, the paths point to pre-resampled data, and so there is no on-the-fly resampling in the inference dataloader.
Please carefully consider the resolution you choose to operate on and adapt the code accordingly.

# Working repo
If you are in the MASI lab, you can access my working github repo here

https://github.com/MASILab/gca_he/tree/master/segmentation/vhe

This repo includes all code fragments etc.


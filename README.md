# This repo enables nucleus/cell classification on intestinal H&E into 14 classes

# Citations
If you use this repo, please cite
- "Data-driven Nucleus Subclassification on Colon H&E using Style-transferred Digital Pathology"

# Pretrained weights
The weights for 5 folds of trained resnets for nucleus classification on virtual H&E can be found here:
**TODO**

# Inference
An example jupyter notebook showing how a pretrained model can be run on a whole slide image (virtual H&E) can be found in this repo
```inference_on_wsi.ipynb```

If you are in MASI lab, the paths are setup so that this inference notebook will run with the example data.

If you are not in MASI lab, change paths accordingly.

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

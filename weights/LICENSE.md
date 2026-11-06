# Licence of the bundled weights

The files in this directory are **not** covered by the MIT License that
applies to the code of this repository (see `../LICENSE`):

- the segmentation checkpoints (`*.pt` in this directory: four families
  of five cross-validation folds each — S, ResNet34 with soft targets;
  V, EfficientNetV2-S with individual targets; V_soft, EfficientNetV2-S
  with soft targets; and V60k, EfficientNetV2-S with individual targets
  trained with a 60k-step horizon),
- the refiner checkpoints (`refiner/*.pt`: ten cross-fit models for the
  default P3 configuration, seeds 20261104 and 20261105, and five for the
  alternate P configuration, seed 20261104; stored in FP16), and
- the component-tree models (`P3_tree.json`, a bagged pair of
  exact-direct LightGBM models, and `P_tree.json`, a single exact-direct
  model: LightGBM text models inside portable JSON, never pickles).

## What these weights are

Each learned file was trained by us for the Solar Filament Segmentation
Challenge 2026 on the competition training data only.  The neural
checkpoints start from an ImageNet-pretrained encoder checkpoint; the
tree model is trained from scratch on hand-computed component features
(no pretrained weights enter it).

- **Training annotations:** the competition training file of MAGFiLO
  (*Manually Annotated GONG Filaments from H-Alpha Observations*,
  Ahmadzadeh et al., Scientific Data 11, 1031 (2024),
  <https://doi.org/10.1038/s41597-024-03876-y>; Harvard Dataverse
  <https://doi.org/10.7910/DVN/J6JNVK>), distributed under
  **CC BY-NC 4.0** (non-commercial use, attribution required). The weights
  are therefore derivatives of CC BY-NC 4.0 material, and that
  non-commercial condition continues to apply to them.  This covers the
  segmentation folds, the refiners (whose targets are means of the
  annotators' masks) and the tree models (whose targets are IoU
  contributions computed against the same annotations).
- **Training images:** H-alpha full-disk images of the NSO Integrated
  Synoptic Program (GONG network). NSO/GONG data are public domain, with
  the acknowledgement reproduced in the repository README required in
  publications.
- **Pretrained initialisation** (neural checkpoints only; the exact
  checkpoint of every file is recorded in `manifest.json` next to this
  file):
  - EfficientNetV2-S (`timm/tf_efficientnetv2_s.in21k_ft_in1k`,
    ImageNet-21k pretrained, fine-tuned on ImageNet-1k): **Apache-2.0**.
    Used by the V, V_soft and V60k segmentation folds (15 files).
  - ResNet34 (`smp-hub/resnet34.imagenet`, torchvision weights): the
    distributor shows the licence as **"other"** (no explicit licence text);
    it is used here as a starting point only, under the competition rules'
    exception for pretrained models whose licence is incompatible with an
    open-source grant.  Used by the five S-family segmentation folds and
    by the fifteen refiner checkpoints (4-channel U-Net encoder).

## Conditions

To the extent that the upstream pretrained checkpoints or the training
data carry non-commercial or attribution conditions, those conditions
continue to apply to the corresponding parts of these weights. In
particular, use of the weights is limited to non-commercial purposes
consistent with CC BY-NC 4.0, with attribution to MAGFiLO and NSO/GONG.
Competition data and images are not included in this repository.

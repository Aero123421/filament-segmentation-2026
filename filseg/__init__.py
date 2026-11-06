"""Solar filament instance segmentation: the code of this repository.

Modules in pipeline order:

- :mod:`filseg.preprocess`  read a 2048x2048 H-alpha image, central 1888x1888 crop, paste back
- :mod:`filseg.model`       rebuild each U-Net from its checkpoint and load the bundled weights
- :mod:`filseg.infer`       test-time augmentation x fold ensemble -> probability map;
  :class:`filseg.infer.FilamentPipeline` runs the configured end-to-end pipeline
- :mod:`filseg.postprocess` probability map -> disjoint instances (threshold, closing,
  connected components, mean-probability gate)
- :mod:`filseg.tree`        probability map -> disjoint instances (multi-threshold
  component tree scored by a LightGBM model, exact antichain selection)
- :mod:`filseg.submission`  instances -> competition CSV rows (COCO RLE), format checks
- :mod:`filseg.metric`      Panoptic Quality exactly as in the official evaluation

Optional, for retraining: :mod:`filseg.prepare` (training cache from the
competition files) and :mod:`filseg.train` (the two-stage training recipe).

Importing the package loads no weights, downloads nothing and does not touch the GPU.
"""

__version__ = "1.0.0"

# Textured-NeuS-utils

This is the experimental repo of [Textured-NeuS](https://github.com/xrr-233/Textured-NeuS).

The functions of this repo include:

1. Preprocess formal raw datasets into acceptable input form of NeuS.
2. Conduct metrics comparison to demonstrate the effect (in progress).

## Details

### Datasets

In our experiment, two types of formal raw datasets can be converted: [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), and [BlendedMVS](https://github.com/YoYo000/BlendedMVS). For DTU, we start from the **SampleSet**, then substitute the **Surfaces** and **Rectified** data to test. For BlendedMVS, at least two components are needed: **BlendedMVS low_res/high-res set** (one of them, and even other BlendedMVG datasets can be applied), as well as **textured meshes** supplementary.

These datasets are very large, and make sure you have enough space to store them. Expected dataset structure is displayed as below:

```
dataset
|---blendedmvs
   (|---dataset_full_res_xx-xx)
    |---dataset_low_res
    |---dataset
|---dtu
    |---SampleSet
        |---Matlab evaluation code
        |---MVS Data
		...
```

### Code (In progress)

The main code is `test.py`. For first-time execution, the external path should be identified to make an initial conversion of raw datasets. For example:

```
baseline_dtu_dataset = DTUDataset('D:/dataset/dtu')
baseline_bmvs_dataset = BlendedMVSDataset('D:/dataset/blendedmvs')
baseline_dtu_dataset.preprocess_dataset()
baseline_bmvs_dataset.preprocess_dataset()
```

After preprocessing, the code will generate the preprocessed models with the structure that NeuS can accept.

*Note:* we use Open3D visualizer (not off-renderer) in the code, so  the rendering of mesh images will create a window, which may be kind of annoying.

We also re-organize the structure of the results of NeuS for convenience of metrics computing.

**Note:** To compute the ChamferL1 distance, please clone the repo https://github.com/ThibaultGROUEIX/ChamferDistancePytorch into our project.

### Metrics

To demonstrate the high fidelity of the exported 3D model, we intend to judge the performance in two aspects, namely, novel view rendering and 3D reconstruction. We compare the qualities of rendered images, as well as rendered mesh images, by comparing the **PSNR, SSIM, and LPIPS** with the baseline images. Simultaneously, we analyze the **Chamfer-L1 Distance** between the processed mesh and baseline mesh.

*Note:* because of the high computing cost (image rendering and Chamfer-L1), scripts in this project should be conducted on powerful devices.

## Environment

The same as [Textured-NeuS](https://github.com/xrr-233/Textured-NeuS).

## Reference

https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf

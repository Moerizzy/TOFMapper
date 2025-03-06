## Introduction

**TOFMapper** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on detecting Trees outside Forest in high resolution aerial images. 


## Major Features

- Segmentation and classification of aerial imagery into four trees outside forest classes (Forest, Patch, Linear, Tree)
- Six trained models [available](https://myshare.uni-osnabrueck.de/d/1926bba15b42484282fc/)
- Can handle large inference images by slicing and stitching them together using overlapping predictions

## Supported Networks

- Vision Transformer

  - [UNetFormer](https://authors.elsevier.com/a/1fIji3I9x1j9Fs) 
  - [DC-Swin](https://ieeexplore.ieee.org/abstract/document/9681903)
  - [BANet](https://www.mdpi.com/2072-4292/13/16/3065)
  
- CNN
 
  - [ABCNet](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
  - [LSKNet](https://doi.org/10.1007/s11263-024-02247-9)
  - [U-Net](https://arxiv.org/abs/1505.04597)
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
Project
├── TOFMapper (code)
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)
├── model_weights (save the model weights trained on your and our TOF data)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── tof
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── val_images (original)
│   │   ├── val_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── test (processed)
│   ├── sites
│   │   ├── site name
│   │   |   ├── TOP (True Orthophoto)     Name: TOP_id.tif
│   │   |   ├── SHP (Reference Shapefile) Name: site_name_TOF.shp
```

## Install

Open the folder **Project** using **Linux Terminal** and create python environment:
```
conda create -n TOFMapper python=3.8
conda activate TOFMapper
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r TOFMapper/requirements.txt
```

## Pretrained Weights of Backbones

[Download](https://myshare.uni-osnabrueck.de/d/f91f2814194a4e708822/)

## Trained Models for TOF Mapping and Classifcation

| Model           | mIoU  | mF1   | Forest IoU | Forest F1 | Patch IoU | Patch F1 | Linear IoU | Linear F1 | Tree IoU | Tree F1 |
|-----------------|-------|-------|------------|-----------|-----------|----------|------------|-----------|----------|---------|
| ABCNet          | 0.709 | 0.821 | **0.971** | 0.944     | 0.532     | 0.695    | 0.739      | 0.850     | 0.622    | 0.767   |
| BANet           | 0.656 | 0.777 | 0.937      | 0.967     | 0.422     | 0.593    | 0.689      | 0.816     | 0.575    | 0.730   |
| DC-Swin         | 0.714 | 0.824 | 0.949      | 0.974     | 0.570     | 0.726    | 0.751      | 0.858     | 0.586    | 0.727   |
| FT-UNetFormer   | **0.739** | **0.843** | 0.952      | **0.975** | **0.606** | **0.754** | **0.774**  | **0.872** | **0.626** | **0.770** |
| LSKNet          | 0.697 | 0.812 | 0.943      | 0.971     | 0.545     | 0.705    | 0.718      | 0.836     | 0.582    | 0.736   |
| U-Net           | 0.484 | 0.618 | 0.804      | 0.891     | 0.151     | 0.263    | 0.527      | 0.690     | 0.456    | 0.626   |

[Download Model Weights](https://myshare.uni-osnabrueck.de/d/1926bba15b42484282fc/)

## Data Preprocessing

### Create Reference Data (optional)
Create your own reference data using this [repository](https://github.com/Moerizzy/Manual_TOF_Detection.git). Be aware that manual refinement will be required!

Create images (masks) in the same size as orthophotos from shapefiles.
```
python TOFMapper/tools/create_masks.py --state "site name" --epsg "EPSG:25833"
```

Define the training, validation and test split (here 90, 5, 5). This will split the data so that all classes are equally distributed. It is set up for 100 files.
```
python TOFMapper/tools/data_statistics.py --state "site name"
```

Copy the files into the designated training, validation and testing folders.
```
python copy_files.py --sites SITE_A SITE_B \
            --dest_train_images "custom/train_images" \
            --dest_train_masks "custom/train_masks" \
            --dest_val_images "custom/val_images" \
            --dest_val_masks "custom/val_masks" \
            --dest_test_images "custom/test_images" \
            --dest_test_masks "custom/test_masks"
```

### Spliting the Data

Generate the training set.
```
python TOFMapper/tools/tof_patch_split.py \
--img-dir "data/tof/train_images" \
--mask-dir "data/tof/train_masks" \
--output-img-dir "data/tof/train/images_1024" \
--output-mask-dir "data/tof/train/masks_1024"\
 --mode "train" --split-size 1024 --stride 1024 \
```
Generate the validation set.
```
python TOFMapper/tools/tof_patch_split.py \
--img-dir "data/tof/val_images" \
--mask-dir "data/tof/val_masks" \
--output-img-dir "data/tof/val/images_1024" \
--output-mask-dir "data/tof/val/masks_1024"\
 --mode "val" --split-size 1024 --stride 1024 \
```
Generate the testing set.
```
python TOFMapper/tools/tof_patch_split.py \
--img-dir "data/tof/test_images" \
--mask-dir "data/tof/test_masks" \
--output-img-dir "data/tof/test/images" \
--output-mask-dir "data/tof/test/masks" \
--mode "val" --split-size 1024 --stride 1024 \
```

## Training

"-c" path of the config, use different **config** to train different models.

```
python TOFMapper/train_supervision.py -c TOFMapper/config/tof/ftunetformer.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

```
python TOFMapper/tof_test.py -c TOFMapper/config/tof/unetformer.py -o fig_results/tof/unetformer --rgb
```

## Inference on Huge Areas

This function takes an image folder and performs inference. It predicts overlapping predictions, which can be adjusted using "-st" and "-ps". These are merged by averaging the probability scores and output as a georeferenced GeoTIF and ShapeFile in the size of the original images.

"i" denotes the input image folder 

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"-st" denotes the stride in pixels for each patch that will be predicted.

"-ps" denotes the patch size that will be predicted

"-b" denoted the batch size


```
python TOFMapper/inference_huge_image.py \
-i data/inference/images \
-c TOFMapper/config/tof/ftunetformer.py \
-o data/inference/mask \
-st 256 \
-ps 1024 \
-b 2 \
```

## Citation

Comming soon!

## Acknowledgement

Many thanks to [Libo Wang](https://github.com/WangLibo1995) for creating [TOFSeg](https://github.com/WangLibo1995/GeoSeg), which served as the foundation for this project.

- [TOFMapper](https://github.com/WangLibo1995/TOFMapper)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

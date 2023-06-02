# Cyclic Learning: Bridging Image-level Labels and Nuclei Instance Segmentation (experiment part 2)

Official implementation of Cyclic Learning: Bridging Image-level Labels and Nuclei Instance Segmentation
The original paper link is here:
[arXiv link](to be update), [TMI link](to be update)
This project provide code for experiments based on hovernet
## Installation

- Our project is developed on Hovernet. We modify some config for our specific task.

- Create an environment meets the requirements as listed in `requirements.txt`

## Data Preparation
- Download the [Monusac dataset](https://pan.baidu.com/s/1ALRjHBQ7LwY-stIW1NzMRA?pwd=mseg) (pwd：mseg) and [cropped Monusac dataset](https://pan.baidu.com/s/1D9F1pLcu2bHwglE1oafmZA?pwd=mseg) (pwd : mseg), and put it in the `Mask_RCNN/datasets/` directory.

- Download the [ccrcc dataset](https://pan.baidu.com/s/1RiuaRxxgXWEa2wNYf58bmw?pwd=mseg)
  (pwd：mseg), and put it in the `Mask_RCNN/datasets/` directory.

- Download the [consep dataset](https://pan.baidu.com/s/1zPPOQI9ZTKpvTlNkePIxmw?pwd=mseg) (pwd：mseg), and put it in the `Mask_RCNN/datasets/` directory.
- Download the positive-and-negative nucleus image [classification dataset](https://pan.baidu.com/s/1CjcIfT2k92gmaLW17noFMw?pwd=mseg) (pwd : mseg) which is obtained by cropping out tile images from TCGA WSI(whole slide image). 
Datasets are organized in the following way:
```bazaar
datasets/
    MyNP/
        negative/
        positive/
    MoNuSACGT/
    MoNuSACCROP/
        stage1_train/
        images/
        masks/
    ccrcccrop/    
        Test/
        Train/
        Valid/
    consepcrop/
        Test/
        Train/
        Valid/
```


## Training
Before training, please download pretrain weights of big nature image datasets, for which we use [COCO pretrain weights](https://cocodataset.org/#home). Remember to change the path in the code.
Training Cyclic Learning on MoNusac dataset:
```bash 
python cyclic.py
```
For ccrcc and consep datasets, please refer to current version and change some paths.
## Citing Cyclic Learning
If you use Cyclic Learning in your work or wish to refer to the results published in this repo, please cite our paper:
```BibTeX

```




# Multi-scale Multi-structure Siamese Network (MMSNet) for Primary Open-angle Glaucoma Prediction

Primary open-angle glaucoma (POAG) is one of the leading causes of irreversible blindness in the United States and worldwide. Although deep learning methods have been proposed to diagnose POAG, these methods all used a single image as input. Differently, the glaucoma specialists compare the follow-up image with the baseline image to determine a glaucomatous eye. To simulate this process, we proposed a siamese network model, POAGNet, to identify POAG from fundus photographs. 

## Datasets

[Ocular Hypertension Treatment Study (OHTS)](https://ohts.wustl.edu/) is one of the largest longitudinal clinical trials in POAG (1,636 participants and 37,399 images) from 22 centers in the United States. The study protocol was approved by an independent Institutional Review Board at each clinical center. Please visit the website to obtain a copy of the dataset.

[Sequential fundus Images for Glaucoma (SIG)](https://github.com/XiaofeiWang2018/DeepGF) contains 3,684 fundus images, of which 153 (4.15%) have POAG.  

## Getting started

### Prerequisites

* python >=3.6
* keras
* tensorflow-gpu = 2.2.0
* sklearn
* pandas
* opencv
* skimage

### Quickstart

```sh
python train.py
```

### Reference



### Acknowledgment

This project was supported by the National Library of Medicine under award number 4R00LM013001. This work was also supported by awards from the National Eye Institute, the National Center on Minority Health and Health Disparities, National Institutes of Health (grants EY09341, EY09307), Horncrest Foundation, awards to the Department of Ophthalmology and Visual Sciences at Washington University, the NIH Vision Core Grant P30 EY 02687, Merck Research Laboratories, Pfizer, Inc., White House Station, New Jersey, and unrestricted grants from Research to Prevent Blindness, Inc., New York, NY.  

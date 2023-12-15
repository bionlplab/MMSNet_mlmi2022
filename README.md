# Multi-scale Multi-structure Siamese Network (MMSNet) for Primary Open-angle Glaucoma Prediction

Primary open-angle glaucoma (POAG) is one of the leading causes of irreversible blindness in the United States and worldwide. Several deep learning methods have been proposed to detect POAG from fundus photographs. All of these studies predict the current glaucomatous status of a patient. In this study, we seek to predict the probability of POAG onset from fundus photos. Such prediction may identify patients appropriate for early treatment. To the best of our knowledge, only two previous works have focused on prediction of future POAG event. In clinical practice,  patients usually have follow-up visits to screen for glaucoma progression. Therefore, the glaucoma specialists compare the follow-up with the baseline image (the image taken at the first visit of a study) to trace the relevant feature. In this study, we used fundus images to predict an eye's progress to POAG (which may never occur) within specific inquired durations from the current visit. The inquired duration was selected in advance (2-year or 5-year), and it was relative to the time when the image was taken, not to the time of the baseline visit.Unlike prior studies, for one eye, the inputs included one fundus image taken at the baseline (first visit), and one image was taken at the current visit (follow-up image). Therefore, our proposed method is suitable for screening patients during follow-up visits. We never need “future images” to screen people. The output was the probability that the time to POAG onset exceeds the inquired duration. To handle the pair of images, we proposed a novel Siamese network model with side output and additional convolution, called multi-scale multi-structure Siamese network (MMSNet), by comparing the differences between two input images.

## Datasets

[Ocular Hypertension Treatment Study (OHTS)](https://ohts.wustl.edu/) is one of the largest longitudinal clinical trials in POAG (1,636 participants and 37,399 images) from 22 centers in the United States. The study protocol was approved by an independent Institutional Review Board at each clinical center. Please visit the website to obtain a copy of the dataset.
 

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
python train_cv_series.py
```

### Reference

Lin M, Liu L, Gorden M, Kass M, Tassel SV, Wang F, Peng Y. Multi-scale Multi-structure Siamese Network (MMSNet) for Primary Open-Angle Glaucoma Prediction. In International Workshop on Machine Learning in Medical Imaging (MLMI). 2022 Sep;13583:436-445. doi: 10.1007/978-3-031-21014-3_45. Epub 2022 Dec 16. PMID: 36656619; PMCID: PMC9844668.

### Acknowledgment

This project was supported by the National Library of Medicine under award number 4R00LM013001. This work was also supported by awards from the National Eye Institute, the National Center on Minority Health and Health Disparities, National Institutes of Health (grants EY09341, EY09307), Horncrest Foundation, awards to the Department of Ophthalmology and Visual Sciences at Washington University, the NIH Vision Core Grant P30 EY 02687, Merck Research Laboratories, Pfizer, Inc., White House Station, New Jersey, and unrestricted grants from Research to Prevent Blindness, Inc., New York, NY.  

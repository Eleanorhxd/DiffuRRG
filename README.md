# DiffuRRG

**This is the data and code for our paper** `DiffuRRG: A Pseudo-Label Guided Conditioned Diffusion Framework for Radiology Report Generation`.


## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.8]
- [PyTorch = 1.8]
- [Torchvision]
- [Pycocoevalcap]

* You can create the environment via conda:
```bash
conda env create --name [env_name] --file env.yml
```




## Datasets
We use two publicly available radiology report generation datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://openi.nlm.nih.gov/faq).

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/).

The training set, test set, and validation set data of the IU X-Ray and MIMIC-CXR datasets are shown in the following table:
| Dataset   |           | IU X-Ray    |           |           | MIMIC-CXR  |           |
|-----------|:---------:|:-----------:|:---------:|:---------:|:----------:|:---------:|
| Dataset   |  TRAIN    |     VAL     |   TEST    |   TRAIN   |    VAL     |   TEST    |
| IMAGE#    |  5,226    |     748     |   1,496   |  368,960  |   2,991    |   5,159   |
| REPORT#   |  2,770    |     395     |    790    |  222,758  |   1,808    |   3,269   |
| PATIENT#  |  2,770    |     395     |    790    |  64,586   |    500     |    293    |
| AVG.LEN   |  37.56    |    36.78    |   33.62   |   53      |   53.05    |   66.4    |


After downloading the datasets, put them in the directory `data`.

## Documentation

```
--data
  │--iu_xray
    │--images
    │--annotation.json
  │--mimic_cxr
    │--images
    │--annotation.json
  
 --files
  │--iu_xray
    │--labels.json
  │--mimic_cxr
    │--labels_all_one.json
    
 --CDFR
  │--ddpm_modules
  │--sasm_modules
    
--models
  │--model.py

--modules
  │--utils.py
  │--visual_extractor.py
  │--dataset.py
  │--dataloaders.py
    .......
  
--src
  │--README.md
  │--train.py
  │--run_iu.sh
  │--run_cxr.sh  
```

## Pseudo Label Generation
You can generate the pseudo label for each dataset by leveraging the automatic labeler chexpert-labeler .  
After generate the pseudo label, put them in the directory `files`.

Our experiments were done on RTX 4090 card with 24GB.

## Train on IU X-Ray

Run `bash run_iu.sh` to train a model on the IU X-Ray data.

## Run on MIMIC-CXR

Run `bash run_mimic.sh` to train a model on the MIMIC-CXR data.

## Test on MIMIC-CXR

Run `bash test_mimic.sh` to train a model on the MIMIC-CXR data.

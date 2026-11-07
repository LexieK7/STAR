# ⭐ STAR: Structured Automated Reporting for Pathology

**STAR (STructured Automated Reporting)** is a unified and highly scalable **vision-language model** designed to automatically generate **structured pathology reports** directly from whole-slide pathology images.

Trained and validated on over 10,000 pathology image–report pairs from multiple medical centers, STAR achieves high accuracy in describing key pathological parameters, including tumor subtype, grade, and invasion depth, across both surgical resection and biopsy samples.

Beyond diagnostic report generation, STAR demonstrates cross-modal scalability by predicting **mismatch repair (MMR) status**, bridging histomorphological analysis with molecular phenotype prediction.  
This framework establishes a generalizable, interpretable, and standardized approach for data-driven pathology reporting, advancing the integration of AI in precision medicine.

---

# ⚙️ How to use our model?

###  Prerequisite

Clone the repo and create the environment:

```
git clone https://github.com/LexieK7/STAR
cd STAR
conda env create -f environment.yml
conda activate star
```

Download the pre-trained [GPT-2](https://huggingface.co/openai-community/gpt2) weights from Hugging Face.


## 📁 Data Preparation

The structure of the dataset is shown as follows:

```
|-- data/ # Annotation preparation
|     |--CRC_report
|            |--TCGA_CRC_TEST.json
|            |--...(Other original annotation files)
|     |--split_ViT-B_32_train.pkl
|     |--split_ViT-B_32_train_tokens.pkl
├── feature/ # Patch features
|     |--WSI_slides
|            |--TCGA-A6-2679-01Z-00-DX1.8df66ef4-d9e5-41db-836d-f0afe46d6b5a.svs
|            |--TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e.svs
|            |--...
|     |--patch_feature
|            |--pt_files
|                   |--TCGA-A6-2679-01Z-00-DX1.8df66ef4-d9e5-41db-836d-f0afe46d6b5a.pt
|                   |--TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e.pt
|                   |--...
|            |--h5_files
|                   |--TCGA-A6-2679-01Z-00-DX1.8df66ef4-d9e5-41db-836d-f0afe46d6b5a.h5
|                   |--TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e.h5
|                   |--...
└── README.md
```

The feature directory stores image features extracted from each WSI, while the data directory contains the corresponding pathology reports.
Reports are saved in a JSON file (e.g., TCGA_CRC_TEST.json), which follows the structured format described below.

```
[
 {
  "caption": "Adenocarcinoma, poorly differentiated, pt2.",
  "image_id": "TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4"
 },
 {
  "caption": "Adenocarcinoma, moderately differentiated, pt4a.",
  "image_id": "TCGA-4N-A93T-01Z-00-DX1.82E240B1-22C3-46E3-891F-0DCE35C43F8B"
 },
 {
  "caption": "Mucinous adenocarcinoma, poorly differentiated, pt3.",
  "image_id": "TCGA-4T-AA8H-01Z-00-DX1.A46C759C-74A2-4724-B6B5-DECA0D16E029"
 },
 ...
 ]
```

The TCGA (The Cancer Genome Atlas) dataset used in this study can be publicly accessed and downloaded from the **Genomic Data Commons (GDC) Data Portal**:  
🔗 [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/)

## WSI Preprocessing

In this work, we adpoted [CLAM](https://github.com/mahmoodlab/CLAM) for preprocessing and feature extraction. For installation guide, we recommend to follow the original [instructions](https://github.com/mahmoodlab/CLAM/blob/master/docs/INSTALLATION.md).


Then, based on the extracted .pt and .h5 features, we can derive the slide-level feature representation by aggregating patch-level features. Finally, the slide-level feature and patch-level feature can be concatenated to form the complete representation of each WSI:

```
python get_wsi_feature.py
python concat_feature.py
```

Integrate the text and features into a single training file：

```
python parse_crc.py --clip_model_type ViT-B/32
```


###  Training


Train:

```
python train_fea_pool.py --data ./data/CRC_report/split_ViT-B_32_train.pkl --out_dir ./report/
```

If you want to generate MMR-related information:

```
python train_fea_pool_mmr.py --data ./data/CRC_report/split_ViT-B_32_train.pkl --out_dir ./report_mmr/
```

### Testing

If you want to test NLP-related metrics:

```
python cocoeval_wsi_pool.py
```

If you want to get the performance of each pathology parameter：
```
python item_all.py
```

### Step-by-Step Tutorial

This tutorial provides an overview of the complete STAR workflow, including environment setup, data preparation, WSI feature extraction, model training, report generation, and performance evaluation.

#### Step 1. Installation

Clone the repository and create the required environment:

```bash
git clone https://github.com/LexieK7/STAR.git
cd STAR

conda env create -f environment.yml
conda activate star
```

Download the required pretrained GPT-2 weights from [Hugging Face](https://huggingface.co/openai-community/gpt2).

#### Step 2. Data Preparation

STAR uses pre-extracted WSI features and corresponding pathology reports as inputs.

The image features are stored in the `feature/` directory, while the pathology report annotations are stored in `data/CRC_report/`.

Each annotation contains an `image_id` identifying the WSI and a `caption` containing the corresponding pathology report.

#### Step 3. WSI Feature Extraction

We follow the CLAM preprocessing pipeline to extract patch-level features from WSIs.

After obtaining the patch-level features, run:

```bash
python get_wsi_feature.py
python concat_feature.py
```

These scripts prepare the slide-level and patch-level representations used by STAR.

Next, integrate the image representations and corresponding report annotations:

```bash
python parse_crc.py --clip_model_type ViT-B/32
```

#### Step 4. Model Training

Train STAR for structured pathology report generation:

```bash
python train_fea_pool.py \
    --data ./data/CRC_report/split_ViT-B_32_train.pkl \
    --out_dir ./report/
```

For the extended MMR-related reporting task:

```bash
python train_fea_pool_mmr.py \
    --data ./data/CRC_report/split_ViT-B_32_train.pkl \
    --out_dir ./report_mmr/
```

#### Step 5. Report Generation

After training, load the trained STAR checkpoint and the corresponding WSI features to generate structured pathology reports.

The generated reports describe key pathological parameters, including histological subtype, grade, depth of invasion, venous invasion, and perineural invasion.

A complete inference example, including checkpoint loading, input configuration, and output files, should be provided here.

#### Step 6. Performance Evaluation

Evaluate report-generation performance using standard NLP metrics:

```bash
python cocoeval_wsi_pool.py
```

Evaluate individual pathological parameters:

```bash
python item_all.py
```

These scripts support evaluation at both the report and individual pathological parameter levels.

### Basic Environment

Linux (Tested on Ubuntu 18.04)

NVIDIA GPU (Tested on GeForce RTX 3090 ) with CUDA 12.2

Python (3.9)

Torch (1.9.0+cu111)

torchvision (0.10.1)

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```

```

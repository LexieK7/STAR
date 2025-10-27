# ⭐ STAR: Structured Automated Reporting for Pathology

**STAR (STructured Automated Reporting)** is a unified and highly scalable **vision-language model** designed to automatically generate **structured pathology reports** directly from whole-slide pathology images.

Trained and validated on over 10,000 pathology image–report pairs from multiple medical centers, STAR achieves high accuracy in describing key pathological parameters, including tumor subtype, grade, and invasion depth, across both surgical resection and biopsy samples.

Beyond diagnostic report generation, STAR demonstrates cross-modal scalability by predicting **mismatch repair (MMR) status**, bridging histomorphological analysis with molecular phenotype prediction.  
This framework establishes a generalizable, interpretable, and standardized approach for data-driven pathology reporting, advancing the integration of AI in precision medicine.

---

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
|     |--pt_files
|            |--TCGA-A6-2679-01Z-00-DX1.8df66ef4-d9e5-41db-836d-f0afe46d6b5a.pt
|            |--TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e.pt
|            |--...
|     |--h5_files
|            |--TCGA-A6-2679-01Z-00-DX1.8df66ef4-d9e5-41db-836d-f0afe46d6b5a.h5
|            |--TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e.h5
|            |--...
└── README.md
```

The feature directory stores image features extracted from each WSI, while the data directory contains the corresponding pathology reports.
Reports are saved in a JSON file (e.g., TCGA_CRC_TEST.json), which follows the structured format described below.

```json
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

## How to use our model?

### 1. Installation

Clone the repo and create the environment:

```
git clone https://github.com/LexieK7/STAR
cd STAR
conda env create -f environment.yml
conda activate star
```

Download pre-trained GPT2 weights.

### 2. Preprocess whole slide images

CLAM(https://github.com/mahmoodlab/CLAM) is used to extract Patch features.

Preprocess data:

```
python get_wsi_feature.py
python concat_feature.py
```
### 3. Training

Extract features:

```
python parse_crc.py 
```

Train:

```
python train_fea_pool.py --data ./data/CRC_report/split_ViT-B_32_train.pkl --out_dir ./report/
```

If you want to generate MMR-related information:

```
python train_fea_pool_mmr.py --data ./data/CRC_report/split_ViT-B_32_train.pkl --out_dir ./report_mmr/
```

### 4. Testing

NLP metrics:

```
python cocoeval_wsi_pool.py
```
OR
```
python cocoeval_wsi_pool_mmr.py
```

If you want to get the performance of each pathology parameter：
```
python item_all.py
```
OR
```
python item_mmr.py
```

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```

```

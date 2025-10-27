# 🧠 STAR: Structured Automated Reporting for Pathology

**STAR (STructured Automated Reporting)** is a unified and highly scalable **vision-language model** designed to automatically generate **structured pathology reports** directly from whole-slide pathology images.

Trained and validated on over **10,000 pathology image–report pairs** from multiple medical centers, STAR achieves high accuracy in describing key pathological parameters, including **tumor subtype**, **grade**, and **invasion depth**, across both surgical resection and biopsy samples.

Beyond diagnostic report generation, STAR demonstrates **cross-modal scalability** by predicting **mismatch repair (MMR) status**, bridging histomorphological analysis with **molecular phenotype prediction**.  
This framework establishes a **generalizable, interpretable, and standardized** approach for data-driven pathology reporting, advancing the integration of AI in precision medicine.

---

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

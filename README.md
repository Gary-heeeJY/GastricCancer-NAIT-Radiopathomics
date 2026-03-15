# GastricCancer-NAIT-Radiopathomics
# Preoperative Prediction of pCR in Gastric Cancer (Chapter 3)

This repository contains the source code for the research: **"Preoperative Prediction of Pathological Complete Response (pCR) in Gastric Cancer based on CT Radiomics and Biopsy Pathology"**.

## 📑 Research Overview
This study proposes a multi-modal fusion framework to predict immunotherapy response (pCR) in gastric cancer patients. By integrating CT imaging, biopsy whole-slide imaging (WSI), and standardized clinical text, we achieve superior predictive performance compared to single-modality models.

## 📁 Repository Structure (Based on Chapter 3)
Following the organization in the dissertation, the code is structured as follows:

- `Data_process/`: LLM-driven text standardization and WSI preprocessing.
- `Models/`: Implementation of the AB-MIL (Attention-based Multi-instance Learning) and Fusion architectures.
- `Engine/`: Training pipelines and cross-validation logic.
- `Configs/`: Hyperparameter settings for different modalities.
- `utils/`: Common tools for data loading and feature extraction.
- `script.py`: The main entry point for model execution.

## 🛠 Clinical Evaluation Tools
The repository includes scripts for generating critical clinical metrics discussed in Section 3:
- **ROC Curves**: Performance comparison across modalities.
- **Calibration Analysis**: Reliability check for probability predictions.
- **Decision Curve Analysis (DCA)**: Clinical utility and net benefit assessment.

## 🚀 Key Technologies
- **AB-MIL Aggregation**: Necessary for biopsy scenarios with limited samples.
- **Multi-modal Fusion**: Progressive gain effect through CT and Pathology integration.
- **Standardized NLP**: LLM-driven preprocessing for medical text.

## ✉️ Contact
For review purposes, this repository is currently anonymized. For technical inquiries, please refer to the contact information provided in the dissertation.

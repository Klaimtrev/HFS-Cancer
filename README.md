# Heuristic Feature Selection Methods for Cancer Prediction

## Overview

This repository presents heuristic feature selection methods applied to cancer prediction using machine learning. The goal is to improve classification performance by selecting the most relevant features from gene expression data. The repository contains implementations of multiple feature selection techniques and evaluates their impact on different machine learning models.

## Features

- **Feature Selection Techniques:** Includes heuristic-based methods to identify the most relevant features.
- **Machine Learning Models:** Evaluates performance using Decision Trees (DT), k-Nearest Neighbors (KNN), Naive Bayes (NB), and Support Vector Machines (SVM).
- **Performance Metrics:** Uses MCC (Matthews Correlation Coefficient) and other evaluation metrics to assess model effectiveness.
- **Dataset:** Works with gene expression datasets for cancer classification.

## Installation

To run the project, ensure you have Python installed along with the necessary dependencies.

```sh
git clone https://github.com/Klaimtrev/Heuristic-Feature-Selection-Methods-for-Cancer-Prediction.git
```
```sh
cd Heuristic-Feature-Selection-Methods-for-Cancer-Prediction
```
```sh
python -m venv venv
```
```sh
venv\Scripts\activate
```
```sh
pip install -r requirements.txt
```

## Usage

1. Go to the CFS folder:
   ```sh
   cd CFSMethod
   ```
2. Run the main file with a dataset:
   ```sh
   python .\main.py filepath
   ```
   For example: 
   ```sh
   python main.py "D:\0_RepoTesting\third\Heuristic-Feature-Selection-Methods-for-Cancer-Prediction\Datasets\AP_Breast_Colon.arff"

   ```

## Directory Structure

```
Heuristic-Feature-Selection-Methods-for-Cancer-Prediction/
├── data/               # Datasets used for training and evaluation
├── src/                # Source code for feature selection and model evaluation
├── results/            # Output results and performance metrics
├── requirements.txt    # Dependencies
├── main.py             # Main script to run the project
└── README.md           # Project documentation
```

## Results

The results include performance comparisons of different feature selection methods and machine learning models. The output files in the `results/` directory summarize the findings.

## Author

- **Diego Minaya**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work is heavily inspired by Zixiao Shen’s implementation of correlation-based feature selection. You can find his original work here: [Correlation-based-Feature-Selection](https://github.com/ZixiaoShen/Correlation-based-Feature-Selection/tree/master).  

## References  
 

This work utilizes publicly available datasets for evaluation: 
| Dataset                                  | Number of Features | Cancer Types            | Samples |
|------------------------------------------|--------------------|-------------------------|---------|
| AP_Colon_Kidney (Stiglic, 2014)         | 10936             | Colon                   | 286     |
|                                          |                    | Kidney                  | 260     |
| AP_Breast_Kidney (Stiglic, 2014)        | 10936             | Breast                  | 344     |
|                                          |                    | Kidney                  | 260     |
| AP_Breast_Ovary (Stiglic, 2014)         | 10936             | Breast                  | 344     |
|                                          |                    | Ovary                   | 198     |
| AP_Breast_Colon (Stiglic, 2014)         | 10936             | Breast                  | 344     |
|                                          |                    | Colon                   | 286     |
| AP_Lung_Kidney (Stiglic, 2014)          | 10936             | Lung                    | 126     |
|                                          |                    | Kidney                  | 260     |
| Gene Expression Cancer UCI (Fiorini, 2016) | 20531             | Breast (BRCA)           | 300     |
|                                          |                    | Lung (LUAD)             | 141     |
|                                          |                    | Colon (COAD)            | 78      |
|                                          |                    | Prostate (PRAD)         | 136     |
|                                          |                    | Kidney (KIRC)           | 146     |
| Cancer Types (Christos Ferles, 2018)    | 972               | Breast (BRCA)           | 878     |
|                                          |                    | Lung (LUAD)             | 162     |
|                                          |                    | Uterine (UCEC)          | 269     |
|                                          |                    | Lung (LUSC)             | 240     |
|                                          |                    | Kidney (KIRC)           | 537     |
| Leukemia (Golub, 1999)                  | 7128              | Leukemia                | 72      |



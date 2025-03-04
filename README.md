# ML Statistical Methods for Classification and Approximation

**Author**: Hon Wa Ng
**Date**: October 2024  

## Overview

This repository contains an implementation of statistical and machine learning methods for classification and approximation problems. The project applies techniques such as decision trees, k-nearest neighbors, and vectorization to classify climate-related textual data.

The dataset is included in this repository under the data/ directory.

## Objectives

- Implement and compare classification models for text-based datasets.
- Explore decision trees and k-nearest neighbors for predictive analysis.
- Evaluate model performance using accuracy metrics.
- Apply feature extraction techniques, such as CountVectorizer.
- Handle missing or non-existent data cases.

## Repository Structure
```bash
ML-STATISTICAL-METHODS-FOR-CLASSIFICATION-AND-APPROXIMATION/
│── data/                              # Dataset storage
│   ├── h1_data/                        # Climate dataset
│   │   ├── DNE_climate.csv              # Dataset (may be missing in some cases)
│   │   ├── exists_climate.csv           # Dataset (exists)
│   │   ├── h1_data.zip                   # Compressed dataset
│
│── docs/                               # Documentation files
│   ├── assignment_questions.pdf         # Original problem statement
│   ├── project_writeup.pdf              # Detailed project report
│
│── src/                                # Source code
│   ├── main.py                          # Core script for classification tasks
│
│── LICENSE                             # MIT License
│── requirements.txt                     # Dependencies for running the project

```

---

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/Edwardnhw/ML-Statistical-Methods-for-Classification-and-Approximation.git
cd ML-Statistical-Methods-for-Classification-and-Approximation

```

### 2. Install Dependencies
Ensure you have Python installed (>=3.7), then run:
```
pip install -r requirements.txt

```

---
## How to Run the Project
Execute the classification script:

```
python src/main.py

```
The script will:

- Load the dataset (exists_climate.csv).
- Perform text classification using Decision Trees and k-NN.
- Output model accuracy and predictions.

---
## Methods Used

1. Data Handling
- Reads and processes text-based climate data.
- Checks for missing data (DNE_climate.csv may be absent).
2. Feature Engineering
- Vectorization using CountVectorizer (word frequency-based feature extraction).
- Handling missing data in case some files do not exist.
3. Machine Learning Models
- Decision Tree Classifier: Constructs hierarchical decision rules.
- k-Nearest Neighbors (k-NN): Classifies based on similar data points.

---

## Results & Analysis

- Model accuracy is evaluated using accuracy_score().
- Comparison of different max_depth values in decision trees.
- Performance trade-offs between complexity and overfitting.

Refer to the project_writeup.pdf in the docs/ folder for detailed results.
---
## License
This project is licensed under the MIT License.




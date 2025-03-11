# Machine Learning Pipeline Repository  
Group name: ML_PseudoPharma
Group member: Vo Nguyen Phat, Nguyen Hoang Quan, Tran Tuan Kiet

## Description  
This repository contains the implementation of a machine learning pipeline, covering data preprocessing, model training, and evaluation.  

In this repository, there are three machine learning implemented models: Decision Tree, Neural Netowrk (MLP Approach), Naive Bayes with Genetic Algorithm Optimizer, and Bayesian Network.


## Repository Structure  
├── data
│ ├── preprocessed/ # Contains preprocessed data files (.npz)
│
├── models
│ ├── trained/ # Contains trained model files (.pkl)
│
├── notebooks
│ ├── *.ipynb # Jupyter notebooks for model training and evaluation
│
├── src
│ ├── data/ # Python scripts for data preprocessing
│ ├── models/ # Python scripts for training model. These files are converted from notebooks in folder **notebooks**


## Installation  
1. **Clone this repository**  
    ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name

2. **Install dependencies**
    ```bash
   pip install -r requirements.txt

## Repository pipeline
📥 Get Data
📂 src/data/preprocess.py
- Downloads raw data
- Preprocesses the data
- Saves the processed data as .npz files in the data/preprocessed/ folder

🎯 Train Models
📂 notebooks/
- Contains Jupyter notebooks for training various models
- Loads .npz files for training
- Performs additional data processing if necessary
- Training models

💾 Save Models
📂 models/trained/
- Stores the best-trained models as .pkl files


## Usage
Data Preprocessing
To generate preprocessed .npz files in data/preprocessed/, run the preprocessing script:
    ```bash
    python src/data/preprocess.py

## Model Training
Execute the Jupyter notebooks in the notebooks/ directory to train models. Trained models will be saved in models/trained/.

## Evaluation
After training, models can be evaluated using the notebooks available in the notebooks/ directory.

## Contact
For questions or support, please create an issue on this repository.

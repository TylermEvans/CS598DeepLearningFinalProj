# CS598DeepLearningFinalProj


## Applying Convolutional Neural Networks to Predict the ICD-9 Codes of Medical Records.

## Link to Original Paper - https://pubmed.ncbi.nlm.nih.gov/33322566/

## Requirements 
The directory has a requirements.txt included, however for posterity these are the libraries needed

numpy==1.20.2
pandas==1.3.5
scikit_learn==1.0.2
torch==1.11.0
Python3==3.7.9 or higher

## Data Download Instruction 

The original data for this project is not available for public use, therefore, MIMIC-3 data was used. The instructions for how to access the data are described on the MIMIC-3 data site. 

The only data files needed are the NOTEEVENTS.csv and DIAGNOSES_ICD.csv files. These must reside in the same directory as main.py 

## How to run

All of the code resides in one file, and kicks off with a main function. The preprocessing and model training + eval all are ran in sequence, and extra data is not written to intermediatery files. It all remains in memory. 

## Results

Unfortunately, the results were not able to be gathered together for this submission. The preprocessing of the data is done and on display in the code, however, the model training and eval had issues, mainly revolving around the collation and processing of the data into a custom dataset. 
More info can be gathered from the project report. 


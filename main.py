# --------------------------------------
######Credit Card Fraud Detection######
######Using Naive Bayes Inference######
# --------------------------------------

"""This is a small example of credit card fault detection using Bayes Theorem.
This case study uses Naive Bayes inferencing in order to detect whether a transaction
is a fraud or not. The data set used in this case is a sample data where all the variables
are anonymized due to confidentiality."""

# ------------------
# Importing packages
# ------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

# ------------------
# Uploading viewing data
# ------------------
# Data Handling: Load CSV
df = pd.read_csv("./data/creditcard.csv")

# get to know list of features, data shape, stat. description.
print(df.shape)
print("First 5 lines:")
print(df.head(5))
print("describe: ")
print(df.describe())
print("info: ")
print(df.info())

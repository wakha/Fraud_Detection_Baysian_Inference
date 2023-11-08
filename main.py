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
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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
# All data is in float and int and there are no null values

# ------------------
# Analysis
# ------------------

# Check Class variables that has 0 value for Genuine transactions and 1 for Fraud
print("Class as pie chart:")
fig, ax = plt.subplots(1, 1)
ax.pie(
    df.Class.value_counts(),
    autopct="%1.1f%%",
    labels=["Genuine", "Fraud"],
    colors=["yellowgreen", "r"],
)
plt.axis("equal")
plt.ylabel("")
plt.show()

# plot Time to see if there is any trend
print("Time variable")
df["Time_Hr"] = df["Time"] / 3600  # convert to hours
print(df["Time_Hr"].tail(5))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
ax1.hist(df.Time_Hr[df.Class == 0], bins=48, color="g", alpha=0.5)
ax1.set_title("Genuine")
ax2.hist(df.Time_Hr[df.Class == 1], bins=48, color="r", alpha=0.5)
ax2.set_title("Fraud")
plt.xlabel("Time (hrs)")
plt.ylabel("# transactions")
plt.show()

# This "Time" feature shows that rate of transactions is picking up during day time.
# But number of transactions have almost similar dependence on time of the day for both the classes.
# So, I believe this feature does not yield any predictive power to distinguish between the two classes.
# But ofcourse I will later test this assumption. For now, I'll keep this feature in data frame. I will drop "Time" but keep "Time_Hr".

df = df.drop(["Time"], axis=1)

# let us check another feature Amount
fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
ax3.hist(df.Amount[df.Class == 0], bins=50, color="g", alpha=0.5)
ax3.set_yscale("log")  # to see the tails
ax3.set_title("Genuine")  # to see the tails
ax3.set_ylabel("# transactions")
ax4.hist(df.Amount[df.Class == 1], bins=50, color="r", alpha=0.5)
ax4.set_yscale("log")  # to see the tails
ax4.set_title("Fraud")  # to see the tails
ax4.set_xlabel("Amount ($)")
ax4.set_ylabel("# transactions")
plt.show()

# interesting to note "all transaction amounts > 10K in Genuine Class only".
# Also this amount feature is not on same scale as principle components.
# So, I'll standardize the values of the 'Amount' feature using StandardScalar and save in data-frame for later use.
df["scaled_Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
df = df.drop(["Amount"], axis=1)

# let us check correlations and shapes of those 25 principal components.
# Features V1, V2, ... V28 are the principal components obtained with PCA.
gs = gridspec.GridSpec(28, 1)
plt.figure(figsize=(6, 28 * 4))
for i, col in enumerate(df[df.iloc[:, 0:28].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(df[col][df.Class == 1], bins=50, color="r")
    sns.distplot(df[col][df.Class == 0], bins=50, color="g")
    ax5.set_xlabel("")
    ax5.set_title("feature: " + str(col))
plt.show()

# For some of the features, both the classes have similar distribution.
# So, I don't expect them to contribute towards classifying power of the model.
# So, it's best to drop them and reduce the model complexity, and hence the chances of overfitting.
# Ofcourse as with my other assumptions, I will later check the validity of above argument.

# Now, it's time to split the data in test set (20%) and training set (80%). I'll define a function for it.


def split_data(df, drop_list):
    df = df.drop(drop_list, axis=1)
    print(df.columns)
    # test train split time
    from sklearn.model_selection import train_test_split

    y = df["Class"].values  # target
    X = df.drop(["Class"], axis=1).values  # features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("train-set size: ", len(y_train), "\ntest-set size: ", len(y_test))
    print("fraud cases in test-set: ", sum(y_test))
    return X_train, X_test, y_train, y_test


# Below is funtion to define classifier and get predictions.
# We can use "predict()" method that checks whether a record should belong to "Fraud" or "Genuine" class.
# There is another method "predict_proba()" that gives the probabilities for each class.
# It helps us to learn the idea of changing the threshold that assigns an instance to class 1 or 0,
# thus we can control precision and recall scores. This would be used to calculate area under ROC.


def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    # fit it to training data
    clf.fit(X_train, y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    # for fun: train-set predictions
    train_pred = clf.predict(X_train)
    print("train-set confusion matrix:\n", confusion_matrix(y_train, train_pred))
    return y_pred, y_pred_prob


# Function to print the classifier's scores
def print_scores(y_test, y_pred, y_pred_prob):
    print("test-set confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("recall score: ", recall_score(y_test, y_pred))
    print("precision score: ", precision_score(y_test, y_pred))
    print("f1 score: ", f1_score(y_test, y_pred))
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:, 1])))


# As I discussed above, some of features have very similar shapes for the two types of transactions,
# so I belive that dropping them should help to reduce the model complexity and thus increase the classifier sensitivity.

# Let us check this with dropping some of the features and checking scores.
# Case-NB-1 : do not drop anything
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test, y_pred, y_pred_prob)

# Case-NB-2 : drop some of principle components that have similar distributions in above plots
drop_list = ["V28", "V27", "V26", "V25", "V24", "V23", "V22", "V20", "V15", "V13", "V8"]
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test, y_pred, y_pred_prob)

# Case-NB-3 : drop some of principle components + Time
drop_list = [
    "Time_Hr",
    "V28",
    "V27",
    "V26",
    "V25",
    "V24",
    "V23",
    "V22",
    "V20",
    "V15",
    "V13",
    "V8",
]
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test, y_pred, y_pred_prob)

# Case-NB-4 : drop some of principle components + Time + 'scaled_Amount'
drop_list = [
    "scaled_Amount",
    "Time_Hr",
    "V28",
    "V27",
    "V26",
    "V25",
    "V24",
    "V23",
    "V22",
    "V20",
    "V15",
    "V13",
    "V8",
]
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test, y_pred, y_pred_prob)

# I would say, Case-NB-4 gives me better model sensitivity (or recall) and precision as compared to Case-NB-1.
#  So dropping some of redundant feature will ofcourse helps to make calculations fast and gain senstivity.

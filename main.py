import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
print("hello")


#1.1 Display first 8 rows, print out all columns in the dataset. List all caragorical variables in the answer.
df = pd.read_csv("drug.csv")

print("The first 8 rows: \n",df.head(5))
print("\nThe columns: \n",df.columns.values)
print("\nThe columns with catagorical variables: \n",df.select_dtypes(include=['object']).head())

#1.2 Check that there is any missing values in each column
nan_values = df.isna()
#print(f"The nan values found are shown here:\n{nan_values.sum()}")

#1.3 Replace all missing values in Sex to be 'M'
df = df.fillna("M")

#1.4 Use the built-in  get_dummies to convert al lcat. variables (exceptt drug) to dummies. What is the size of the DF after Transforming?
dfC = pd.get_dummies(df.drop("Drug",axis = 1))
dfC = pd.concat([dfC,df.Drug],axis = 1)
print("\nSize of catagorical variable dummy dataframe is: ", dfC.shape)

#transform all labels drugy to 1, else 0. Transform the type of the column into int instead of str
dfC.Drug.replace({"DrugY":1},inplace = True)
dfC.loc[dfC.Drug != 1,['Drug']] = 0
dfC.Drug = dfC.Drug.astype('int')

#Baseline accuracy for this classification problem?

#Lets assume our baseline model is a model which chooses "DrugY" for each variable.
#The baseline accuracy would be then found by comparing the occurances of DrugY against the total dataset.
#This is just the mean.
baseline = dfC.Drug.mean()
print(f"\nThe baseline accuracy is equal to: {baseline}")

#Question 2:
y = dfC.Drug.values
X = dfC.drop('Drug',axis='columns').values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)

print(f"In the training data we have {y_train.sum()} patients taking DrugY. \nIn the test data we have {y_test.sum()}.")

#Question 3:
#3.1Take age and na_to_k out of X_train.

age_na_train = np.c_[X_train[:,0],X_train[:,1]] #Takes age and Na_to_K values
age_na_test = np.c_[X_test[:, 0], X_test[:, 1]] #Does same but for the test

drugLR = LogisticRegression(penalty='none')
drugLR.fit(age_na_train,y_train)

print(f"Intercept:\n {drugLR.intercept_.round(3)} \nCoefficients:\n {drugLR.coef_.round(3)}")

#3.2: Compute Accuracy, Precision, Sensitivity, and Specificity.


def compute_performance_classifiers(ypred,y,classes):
    #First, get true positives, true negatives, false positives, and false negatives:
    tp = sum(np.logical_and(ypred == classes[1], y == classes[1]))
    tn = sum(np.logical_and(ypred == classes[0], y == classes[0]))
    fp = sum(np.logical_and(ypred == classes[1], y == classes[0]))
    fn = sum(np.logical_and(ypred == classes[0], y == classes[1]))
    print(f"True Positives: {tp} \nTrue Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}")
    #using these, we calculate the classifiers.
    #Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    # Precision
    precision = tp / (tp + fp)
    # "Of all the + in the data, how many do I correctly label?"
    recall = tp / (tp + fn)
    # Sensitivity
    sensitivity = recall
    # Specificity
    specificity = tn / (fp + tn)
    # Print results
    print(f"Accuracy: {round(acc,4)} Recall: {round(recall,4)} Sensitivity: {round(sensitivity,4)} Specificity: {round(specificity,4)}")

#Lets get predictions for y
y_train_pred = drugLR.predict(age_na_train)
#Lets get performance
print(drugLR.classes_)
compute_performance_classifiers(y_train_pred,y_train,drugLR.classes_)
print("Baseline: ",baseline)

#Use threshold to do that now
threshold = 0.5
y_test_prob = drugLR.predict_proba(age_na_test)
y_prob = drugLR.classes_[(y_test_prob[:,1]>threshold).astype(int)]
compute_performance_classifiers(y_prob,y_test,drugLR.classes_)
print(f"Baseline Accuracy: {baseline}")


#Question 4: Two logistic reg. using C=0.1 and C=1 to training, use all var. in data.
fulltestLR1 = LogisticRegression(C=0.1)
fulltestLR1.fit(X_train,y_train)
print("\n\nFull Variable Test with C = 0.1:")
print(f"Intercept:\n {fulltestLR1.intercept_.round(3)} \nCoefficients:\n {fulltestLR1.coef_.round(3)}")

fulltestLR2 = LogisticRegression(C=1)
fulltestLR2.fit(X_train,y_train)
print("\n\nFull Variable Test with C = 1:")
print(f"Intercept:\n {fulltestLR2.intercept_.round(3)} \nCoefficients:\n {fulltestLR2.coef_.round(3)}")

#Question 5: Compute 4 label-based criteria from 3.2.

#Find y_prob's for both
print("Computing Performance Classifiers for Full Test with C = 0.1")
y_full_test_prob_1 = fulltestLR1.predict_proba(X_test)
y_full_prob_1 = fulltestLR1.classes_[(y_full_test_prob_1[:,1]>threshold).astype(int)]
compute_performance_classifiers(y_full_prob_1,y_test,fulltestLR1.classes_)
print(f"Baseline Accuracy: {baseline}")


print("\n\nComputing Performance Classifiers for Full Test with C = 1")
y_full_test_prob_2 = fulltestLR2.predict_proba(X_test)
y_full_prob_2 = fulltestLR2.classes_[(y_full_test_prob_2[:,1]>threshold).astype(int)]
compute_performance_classifiers(y_full_prob_2,y_test,fulltestLR2.classes_)
print(f"Baseline Accuracy: {baseline}")


#Question 6: PRedict the class by making a sigmoid function
def sigmoid(x):
    return 1 / (1+np.exp(-x))
def class_predictor(X,intercept,coeff):
    #calculate linear combination
    z = intercept + X@coeff.T
    pred_probabilities = sigmoid(z)
    class_pred = (pred_probabilities > 0.5).astype(int)
    return class_pred


age_na_test_class_pred = class_predictor(age_na_test,drugLR.intercept_,drugLR.coef_)
full_test_1_class_pred = class_predictor(X_test,fulltestLR1.intercept_,fulltestLR1.coef_)
full_test_2_class_pred = class_predictor(X_test, fulltestLR2.intercept_, fulltestLR2.coef_)

print(age_na_test_class_pred[:5].T,full_test_1_class_pred[:5].T,full_test_2_class_pred[:5].T)
print(drugLR.predict(age_na_test)[:5],fulltestLR1.predict(X_test)[:5],fulltestLR2.predict(X_test)[:5])

#Question 7: ROC
def roc_curve_get(y_test,y_test_prob,pos_label,):
    print(y_test_prob)
    false_pos_rate, true_pos_rate, _ = roc_curve(y_test,y_test_prob,pos_label=pos_label)
    return false_pos_rate, true_pos_rate, _
fpr_i,tpr_i,_ = roc_curve_get(y_test,y_prob,drugLR.classes_[1])
fpr_1,tpr_1,_ = roc_curve_get(y_test,y_full_prob_1,fulltestLR1.classes_[1])
fpr_2, tpr_2, _ = roc_curve_get(y_test,y_full_prob_2,fulltestLR2.classes_[1])
fig, ax = plt.subplots()
ax = sns.lineplot(x=fpr_i,y=tpr_1,label="Initial Two-Var Fit")
ax2 = sns.lineplot(x=fpr_1,y=tpr_1,label="Full Var Fit w/ C=0.1")
ax3 = sns.lineplot(x=fpr_2, y=tpr_2, label="Full Var Fit w/ C=1")

auci = auc(fpr_i,tpr_i)
auc1 = auc(fpr_1,tpr_1)
auc2 = auc(fpr_2, tpr_2)
plt.title("ROC Curve of Three Classifiers")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
plt.annotate(f"Auc Initial = {round(auci,4)}\nAuc w/ C=0.1 = {round(auc1,4)}\nAuc w/ C=1 = {round(auc2,4)}",xy=(.5,.4))

plt.show()

#Easily believe that the C=0.1 has best chance due to area under graph.

#Question 8:
from sklearn.datasets import fetch_openml # a helper function to download popular datasets
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist["data"]
y = mnist["target"]
y = y.astype(np.uint8)
#y =  pd.DataFrame(mnist["target"])
#y = pd.get_dummies(y).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=11)

SGD_Classifier = SGDClassifier(max_iter=2000,tol=1e-3,random_state=11,n_jobs=6)
SGD_Classifier.fit(X_train,y_train)

SGD_Classifier.predict(X[0])
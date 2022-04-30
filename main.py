from sklearn import datasets
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

st.title("Comparing different Classification ML models")

st.write("""
There are couple of different classifiers available but which one is highly accurate?\n
Select your desired classifier and the dataset you want to work on! You may adjust the different parameter value as well.
""")
dataset_name=st.sidebar.selectbox("Select the dataset",("Iris dataset","Breast cancer dataset", "Digits dataset", "Wine dataset"))
st.header(dataset_name)
classifier_name=st.sidebar.selectbox("Select the classifier",("KNN", "SVM", "Naïve Bayes", "Decision tree", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name=="Iris dataset":
        data=datasets.load_iris()
    elif dataset_name=="Breast cancer dataset":
        data=datasets.load_breast_cancer()
    elif dataset_name=="Digits dataset":
        data=datasets.load_digits()    
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x, y

x, y=get_dataset(dataset_name)
st.write("Shape of dataset: ",x.shape)
st.write("Number of classes: ",len(np.unique(y)))

def add_parameter_ui(clf_name):
    prmtr=dict()
    if clf_name=="KNN":
        k=st.sidebar.slider("k",1,15)
        prmtr["k"]=k
    elif clf_name=="SVM":
        C=st.sidebar.slider("C", 0.01, 10.0)
        prmtr["C"]=C
    elif clf_name=="Naïve Bayes":
        var_smoothing= st.sidebar.slider("var_smoothing", 2.71, 9.00) 
        prmtr["var_smoothing"]=var_smoothing  
    elif clf_name=="Decision tree":
        max_depth=st.sidebar.slider("max_depth", 1, 10)
        prmtr["max_depth"]=max_depth
    else:
        max_depth=st.sidebar.slider("max_depth", 2, 15)
        n_estimators=st.sidebar.slider("n_estimators", 1, 100)
        prmtr["max_depth"]=max_depth
        prmtr["n_estimators"]=n_estimators
    return prmtr

prmtr=add_parameter_ui(classifier_name)

def get_classifier(clf_name, prmtr):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=prmtr["k"])
    elif clf_name=="SVM":
        clf=SVC(C=prmtr["C"])
    elif clf_name=="Naïve Bayes":
        clf=GaussianNB(var_smoothing=prmtr["var_smoothing"])    
    elif clf_name=="Decision tree":
        clf=DecisionTreeClassifier(max_depth=prmtr["max_depth"])      
    else:
        clf=RandomForestClassifier(n_estimators=prmtr["n_estimators"], max_depth=prmtr["max_depth"], random_state=1234)
    return clf

clf=get_classifier(classifier_name, prmtr)

#Training the ML model
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=2)

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

acc=accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy(in %) = {acc*100}")

#Plot
pca=PCA(2)
x_pojected=pca.fit_transform(x)

x1=x_pojected[:, 0]
x2=x_pojected[:, 1]
fig=plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Independent value")
plt.ylabel("Dependent value")
plt.colorbar()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.write("© 2022. Nishayan Debnath. All Rights Reserved.")
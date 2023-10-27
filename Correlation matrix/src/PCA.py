import numpy as np
import pandas as pd  # to load the dataframe
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns  # to plot the heat maps
from sklearn import datasets

def PCA_tmj(data):

    data = data.drop(columns=['Unnamed: 0'])
    data = data.drop(columns=['involvementstatus'])
    data = data.drop(columns=['visitationdate'])
    data = data.drop(columns=['sex'])
    data = data.drop(columns=['type'])
    data = data.drop(columns=['studyid'])
    #data = data.drop(columns=['tractionleft'])
    #data = data.drop(columns=['tractionright'])

    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(data))  # scaling the data

    # fig, ax = plt.subplots(figsize=(32, 24))
    # sns.heatmap(scaled_data.corr())
    # plt.tight_layout()
    # plt.savefig("withoutPCA", dpi=300)

    # Applying PCA
    # Taking no. of Principal Components as 3

    pca = PCA(n_components=50)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca)
    # data_pca.head()

    fig, ax = plt.subplots(figsize=(32, 24))
    sns.heatmap(data_pca.corr())
    plt.tight_layout()
    plt.savefig("withPCA", dpi=300)

    print("Done!")

def PCA_iris():

    # Load the Dataset
    iris = datasets.load_iris()
    # convert the dataset into a pandas data frame
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    # display the head (first 5 rows) of the dataset
    df.head()

    # Standardize the features
    # Create an object of StandardScaler which is present in sklearn.preprocessing
    scalar = StandardScaler()
    scaled_data = pd.DataFrame(scalar.fit_transform(df))  # scaling the data

    # fig, ax = plt.subplots(figsize=(32, 24))
    # sns.heatmap(scaled_data.corr())
    # plt.tight_layout()
    # plt.savefig("withoutPCA", dpi=300)

    # Applying PCA
    # Taking no. of Principal Components as 3
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    data_pca = pca.transform(scaled_data)
    data_pca = pd.DataFrame(data_pca)
    data_pca.head()

    fig, ax = plt.subplots(figsize=(32, 24))
    sns.heatmap(data_pca.corr())
    plt.tight_layout()
    plt.savefig("withPCA", dpi=300)

    print("Done!")

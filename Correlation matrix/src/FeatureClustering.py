import numpy as np
from sklearn import datasets, cluster

def feature_clustering():

    digits = datasets.load_digits()
    images = digits.images
    X = np.reshape(images, (len(images), -1))
    agglo = cluster.FeatureAgglomeration(n_clusters=32)
    agglo.fit(X)
    X_reduced = agglo.transform(X)
    X_reduced.shape

    print("Done!")
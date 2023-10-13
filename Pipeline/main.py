import pipeline as p

# Data manipulation
from seaborn import load_dataset
import numpy as np
import pandas as pd
from helper import calculate_roc_auc
pd.options.display.precision = 4
pd.options.mode.chained_assignment = None

# Machine learning pipeline
from sklearn.model_selection import train_test_split
from pipeline import FeatureExtractor, Imputer, CardinalityReducer, Encoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# https://github.com/zluvsand/ml_pipeline
if __name__ == '__main__':

    # Load data
    columns = ['alive', 'class', 'embarked', 'who', 'alone', 'adult_male']
    df = load_dataset('titanic').drop(columns=columns)
    df['deck'] = df['deck'].astype('object')
    print(df.shape)
    df.head()

    SEED = 42
    TARGET = 'survived'
    FEATURES = df.columns.drop(TARGET)

    NUMERICAL = df[FEATURES].select_dtypes('number').columns
    print(f"Numerical features: {', '.join(NUMERICAL)}")

    CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))
    print(f"Categorical features: {', '.join(CATEGORICAL)}\n")

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=TARGET), df[TARGET],
                                                        test_size=.2, random_state=SEED,
                                                        stratify=df[TARGET])

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    pipe = Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('cat_imputer', Imputer(CATEGORICAL)),
        ('cardinality_reducer', CardinalityReducer(CATEGORICAL, threshold=0.1)),
        ('encoder', Encoder(CATEGORICAL)),
        ('num_imputer', Imputer(NUMERICAL, method='mean')),
        ('feature_selector', RFE(LogisticRegression(random_state=SEED, max_iter=500), n_features_to_select=8)),
        ('model', LogisticRegression(random_state=SEED, max_iter=500))
    ])

    pipe.fit(X_train, y_train)
    print(f"Train ROC-AUC: {calculate_roc_auc(pipe, X_train, y_train):.4f}")
    print(f"Test ROC-AUC: {calculate_roc_auc(pipe, X_test, y_test):.4f}")

    top_features = pipe['feature_selector'].feature_names_in_[pipe['feature_selector'].support_]
    print(f"Top {len(top_features)} features: {', '.join(top_features)}")

    print("Done!")


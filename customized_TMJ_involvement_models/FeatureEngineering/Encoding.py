from Utils import Report as r
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from matplotlib import pyplot
from keras import backend as K

class EntityEmbeddingTransformer:
    def __init__(self, target, embeddingList, config):
        self.target = target
        self.embeddingList = embeddingList
        self.config = config

    def fit(self, data, y: None):
        return self

    def transform(self, data, y=None):

        if self.config['do_embedding']:

            for idx, featureEm in enumerate(self.embeddingList):
                string = featureEm
                data = doEmbedding(data, featureEm, self.target, string, self.config['embedding_epochs'], self.config['lag_features'])

        else:
            path = "Embeddings/embeddedFeatures.csv"
            data = pd.read_csv(path)

        r.write_to_report("encoding", "entity embedding")
        r.write_to_report("encoded data size", f"{data.shape}")

        return data

def doEmbedding(data, featureEm, target, embeddingName, epochs, lag_features):
    print(data.isna().sum())
    embeddingData = data.copy()
    embeddingData = embeddingData.fillna(0)

    y = embeddingData[target]

    if lag_features:
        print("Lag features gør noget")
        # X = embeddingData.drop(columns=[target, "ID", "sex", "previousstatus"], axis=1).astype(float)
    else:
        X = embeddingData.drop(columns=[target, "ID", "sex"], axis=1).astype(float)

    # evt. noget stratify på
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    print(f'The training dataset has {X_train.shape[0]} records and {X_train.shape[1]} columns')
    print(f'The testing dataset has {len(X_test)} records')

    print(X_train.head())
    print(y_train.head())

    # remaining cols
    if lag_features:
        print("lag features gøøør noget")
        # remainfeatures = data.drop(columns=[featureEm, target, "ID", "sex", "previousstatus"], axis=1)
    else:
        remainfeatures = data.drop(columns=[featureEm, target, "ID", "sex"], axis=1)

    numeric_cols = remainfeatures.columns
    X_train[numeric_cols] = X_train[numeric_cols].astype(np.float32)
    X_test[numeric_cols] = X_test[numeric_cols].astype(np.float32)

    print(numeric_cols)
    input_numeric = Input(shape=(len(numeric_cols),))
    emb_numeric = input_numeric

    # Input dimension of the categorical variable
    input_cat = Input(shape=(1,))
    # Output dimension of the categorical entity embedding (here we want 1 for complexity)
    cat_emb_dim = 1
    test = X[featureEm].values.max()
    n_unique_cat = int(test)+1
    # Embedding Layer
    emb_cat = Embedding(input_dim=n_unique_cat, output_dim=cat_emb_dim, name="embedding_cat")(input_cat)
    # Reshaping
    emb_cat = Reshape(target_shape=(cat_emb_dim,))(emb_cat)

    input_data = [input_cat, input_numeric]
    emb_data = [emb_cat, emb_numeric]

    # Concatenate layer (concatenating a list of inputs)
    model_data = Concatenate()(emb_data)

    # Concatenated data passed in dense layer
    # Dense layer with 10 neurons and relu activation function
    model = Dense(10, activation='relu', kernel_initializer='he_uniform')(model_data)
    # Dense layer with 5 neurons and relu activation function
    model = Dense(5, activation='relu', kernel_initializer='he_uniform')(model)
    # Dense layer with 2 neurons and relu activation function
    model = Dense(2, activation='relu', kernel_initializer='he_uniform')(model)

    outputs = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=input_data, outputs=outputs, name=embeddingName)
    print(model.summary())

    string = embeddingName
    string += '.png'

    #plot_model(model, show_shapes=True, show_layer_names=True, to_file="Embeddings/"+string)
    #Image(retina=True, filename="Embeddings/"+string)

    # Compile the model and set up early stopping
    opt = SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[f1_score])   # or 'accuracy'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
    history = model.fit(
        [X_train[featureEm], X_train[numeric_cols]],
        y_train,
        validation_data=([X_test[featureEm], X_test[numeric_cols]], y_test),
        epochs=epochs,
        callbacks=[es]
    )

    # Evaluate the model
    _, train_acc = model.evaluate([X_train[featureEm], X_train[numeric_cols]], y_train, verbose=0)
    _, test_acc = model.evaluate([X_test[featureEm], X_test[numeric_cols]], y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # Plot loss during training
    # pyplot.subplot(211)
    # pyplot.title('Loss')
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(history.history['val_accuracy'], label='test')
    # pyplot.legend()
    # pyplot.show()

    cat_encoder = {}
    unique_cat = np.unique(data[featureEm])
    for i in range(len(unique_cat)):
        cat_encoder[unique_cat[i]] = i

    # Getting weights from the embedding layer
    cat_emb_df = pd.DataFrame(model.get_layer('embedding_cat').get_weights()[0]).reset_index()
    # Adding prefix to the embedding
    cat_emb_df = cat_emb_df.add_prefix('emb_')
    # Putting the categorical encoder dictionary into a dataframe
    cat_encoder_df = pd.DataFrame(cat_encoder.items(), columns=[featureEm, 'emb_index'])
    # Merging data to append the category name
    cat_emb_df = pd.merge(cat_encoder_df, cat_emb_df, how='inner', on='emb_index')
    print(cat_emb_df.head())

    csvName = 'Embeddings/'
    csvName += embeddingName
    csvName += '.csv'

    # saving embedding results
    cat_emb_df.to_csv(csvName, index=False)

    X_emb = pd.merge(data, cat_emb_df, left_on=featureEm, right_on=featureEm, how='inner').drop(
        ['emb_index', featureEm], axis=1)
    X_emb[featureEm] = X_emb['emb_0']
    X_emb = X_emb.drop('emb_0', axis=1)

    X_emb.to_csv('Embeddings/embeddedFeatures.csv', index=False)

    return X_emb


"""
Method implemented based on this thread: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
"""
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1
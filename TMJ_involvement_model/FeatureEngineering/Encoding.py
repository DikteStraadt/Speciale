import pandas as pd
import Report as r
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model
from IPython.display import Image
from matplotlib import pyplot

class OneHotEncode:

    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        new_df = data

        for column_name in self.columns_to_encode:
            nominal_encoded_column = pd.get_dummies(data[column_name].astype(str), prefix=column_name)
            new_df.drop(column_name, axis=1, inplace=True)
            new_df = new_df.join(nominal_encoded_column)

        r.write_to_report("encoding", "one hot")
        print("Data encoded")

        return new_df

class EntityEmbeddingTransformer:
    def __init__(self, columns_to_encode, target):
        self.columns_to_encode = columns_to_encode
        self.target = target

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        for idx, x in enumerate(self.columns_to_encode):
            string = "entityEmbedding_" + str(idx)
            data = doEmbedding(data, x, self.target, string)

        return data

def doEmbedding(data, featureEm, target, embeddingName):
    print(data.isna().sum())
    embeddingData = data.copy()
    embeddingData[target].replace(2.0, 1.0)
    y = embeddingData[target]
    X = embeddingData.drop(target, axis=1).astype(float)

    # evt. noget stratify på
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    print(f'The training dataset has {X_train.shape[0]} records and {X_train.shape[1]} columns')
    print(f'The testing dataset has {len(X_test)} records')

    print(X_train.head())
    print(y_train.head())

    # remaining cols
    remainfeatures = data.drop(columns=[featureEm, target], axis=1)
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
    n_unique_cat = len(np.unique(X_train[featureEm]))
    # Embedding Layer
    emb_cat = Embedding(input_dim=n_unique_cat, output_dim=cat_emb_dim, name="embedding_cat")(input_cat)
    # Reshaping
    emb_cat = Reshape(target_shape=(cat_emb_dim, )) (emb_cat)

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

    plot_model(model, show_shapes=True, show_layer_names=True, to_file=string)
    Image(retina=True, filename=string)

    # Compile the model and set up early stopping
    opt = SGD(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(
        [X_train[featureEm], X_train[numeric_cols]],
        y_train,
        validation_data = ([X_test[featureEm], X_test[numeric_cols]], y_test),
        epochs=1000
    )


    # Evaluate the model
    _, train_acc = model.evaluate([X_train[featureEm], X_train[numeric_cols]], y_train, verbose=0)
    _, test_acc = model.evaluate([X_test[featureEm], X_test[numeric_cols]], y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # Plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

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

    csvName = embeddingName
    csvName += '.csv'

    # saving embedding results
    cat_emb_df.to_csv(csvName, index=False)

    X_emb = pd.merge(data, cat_emb_df, left_on=featureEm, right_on=featureEm, how='inner').drop(['emb_index', featureEm], axis=1)
    X_emb[featureEm] = X_emb['emb_0']
    X_emb = X_emb.drop('emb_0', axis=1)

    X_emb.to_csv('embeddedFeatures.csv', index=False)

    return X_emb
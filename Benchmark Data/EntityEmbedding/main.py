# Used for Data processing
import pandas as pd
import numpy as np

# Used for train test split
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Used for Model
from sklearn.ensemble import RandomForestRegressor

# Visualization
import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)}) # Setting figure size
import matplotlib.pyplot as plt
# Visualize neural network model structure
from keras.utils import plot_model
from IPython.display import Image
from math import sqrt, ceil

# Deep learning model
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, Embedding
from tensorflow.keras.models import Model, load_model
from keras.callbacks import EarlyStopping
import datetime as dt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv('Consumo_cerveja.csv', parse_dates=["Data"], decimal=",")
    df.columns = ["Date", "Avg Temp", "Min Temp", "Max Temp", "Rainfall", "Weekend", "Beer Consumption"]
    print(df.head())

    after_drop_data = df.drop(range(365, 941))
    print(after_drop_data.tail())

    date_data = pd.DataFrame(after_drop_data["Date"])
    print(date_data.head())

    after_drop_data["Week Day"] = after_drop_data["Date"].dt.day_name()

    #https://towardsdatascience.com/enhancing-categorical-features-with-entity-embeddings-e6850a5e34ff

    # Countplot of distribution of days
    ax = sns.countplot(after_drop_data["Week Day"])
    ax.bar_label(ax.containers[0])
    ax.tick_params(axis='x', rotation=90)
    plt.show()

    # Features
    X = after_drop_data.iloc[:, 1:].copy().drop(['Weekend','Beer Consumption'], axis=1)

    # Target
    y =   [float(beer) for beer in after_drop_data['Beer Consumption']]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Checking the number of records in training and testing dataset
    print(f'The training dataset has {X_train.shape[0]} records and {X_train.shape[1]} columns')
    print(f'The testing dataset has {len(X_test)} records')


    # Input list for the training data
    input_list_train = []

    # Input list for the testing data
    input_list_test = []

    # Categorical encoder in dictionary format
    cat_encoder = {}

    # Unique values for the categorical variable
    unique_cat = np.unique(X_train['Week Day'])
    print(f'There are {len(unique_cat)} unique week days in the training dataset. \n')


    # Encoding of the categorical variable
    for i in range(len(unique_cat)):
        cat_encoder[unique_cat[i]] = i

    # Appending the values to the input list
    input_list_train.append(X_train['Week Day'].map(cat_encoder).values)
    input_list_test.append(X_test['Week Day'].map(cat_encoder).values)

    # Taking a look at the data
    print('input_list_train: ', input_list_train)
    print('input_list_test: ', input_list_test)


    # Number of unique values in the categorical col
    n_unique_cat = len(unique_cat)

    # Input dimension of the categorical variable
    input_cat = Input(shape=(1,))

    # Output dimension of the categorical entity embedding
    cat_emb_dim = ceil(sqrt(n_unique_cat))

    # Embedding layer
    emb_cat = Embedding(input_dim=n_unique_cat, output_dim=cat_emb_dim, name="embedding_cat")(input_cat)
    print(emb_cat)
    # Reshaping

    emb_cat = Reshape(target_shape=(cat_emb_dim, )) (emb_cat)
    print(emb_cat)

    # list of numerical columns
    numeric_cols = ['Avg Temp', 'Min Temp', 'Max Temp', 'Rainfall']
    # Appending numerical values to the training and testing list

    input_list_train.append(X_train[numeric_cols].values)
    input_list_test.append(X_test[numeric_cols].values)




    #  can see that both datasets contains two arrays each (categorial encoders + numerical variable values)
    print('input_list_train: ', input_list_train)
    print('input_list_test: ', input_list_test)


    # Input dimension of the numeric variables
    input_numeric = Input(shape=(len(numeric_cols),))

    # Output dimension of the numeric variables
    emb_numeric = input_numeric

    # Input data dimensions
    input_data = [input_cat, input_numeric]


    # Embedding dimensions
    emb_data = [emb_cat, emb_numeric]


    # Concatenate layer (concatenating a list of inputs)
    model_data = Concatenate()(emb_data)

    # Concatenated data passed in dense layer
    # Dense layer with 10 neurons and relu activation function
    model = Dense(10, activation='relu')(model_data)
    # Dense layer with 5 neurons and relu activation function
    model = Dense(5, activation='relu')(model)
    # Dense layer with 2 neurons and relu activation function
    model = Dense(2, activation='relu')(model)

    # Obs when output is discrete (softmax)
    outputs = Dense(1, activation='linear')(model)

    # Using Model to group layers into an object with training and inference features
    nn = Model(inputs=input_data, outputs=outputs, name='Entity_embedding_model_keras')

    print(nn.summary())

    plot_model(nn, show_shapes=True, show_layer_names=True, to_file='Entity_embedding_model_keras.png')
    Image(retina=True, filename='Entity_embedding_model_keras.png')

    # Compile model (obs that this is a regression problem)
    nn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # Set up early stopping
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=50,
                       restore_best_weights=True)

    # Fit the model
    history = nn.fit(input_list_train,
                     tf.convert_to_tensor(y_train),
                     validation_data=(input_list_test, tf.convert_to_tensor(y_test)),
                     epochs=1000,
                     batch_size=64,
                     callbacks=[es])

    #print(history)

    # Summarizing history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    # Summarizing history for MAE
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Mean Absolute Error (MAE)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()


    # Making a prediction
    y_test_predict = nn.predict(input_list_test)

    # Changing the predictions from 2D to 1D
    #print(y_test_predict)

    # Visualization
    #ax = sns.scatterplot(y_test, y_test_predict)
    #plt.show()

    # Metrics
    MSE = mean_squared_error(y_test, y_test_predict)
    print(f'The MSE for the model is {MSE:.2f}')
    MAE = mean_absolute_error(y_test, y_test_predict)
    print(f'The MAE for the model is {MAE:.2f}.')
    R2 = r2_score(y_test, y_test_predict)
    print(f'The R-squared for the model is {R2:.2f}.')
    MAPE = mean_absolute_percentage_error(y_test, y_test_predict)
    print(f'The MAPE for the model is {MAPE:.2f}.')

    # Getting weights from the embedding layer
    cat_emb_df = pd.DataFrame(nn.get_layer('embedding_cat').get_weights()[0]).reset_index()

    # Adding prefix to the embedding names
    cat_emb_df = cat_emb_df.add_prefix('cat_')

    # putting the categorical encoder dictionary into a dataframe
    cat_encoder_df = pd.DataFrame(cat_encoder.items(), columns=['cat', 'cat_index'])

    # Merging data to append the category name
    cat_emb_df = pd.merge(cat_encoder_df, cat_emb_df, how='inner', on='cat_index')

    print(cat_emb_df.head())

    # Saving embedding results
    cat_emb_df.to_csv('cat_embedding_keras.csv', index=False)

    # Save model
    nn.save("cat_embedding_keras.keras")

    # Loading model
    loaded_nn = load_model("cat_embedding_keras.keras")



    # ---- BASELINE MODEL ---- #
    base_cols = ['Avg Temp', 'Min Temp', 'Max Temp', 'Rainfall']

    # Initiate the model
    base_rf = RandomForestRegressor()

    # Fit the model
    base_rf.fit(X_train[base_cols], y_train)

    # Make predictions
    base_y_test_prediction = base_rf.predict(X_test[base_cols])

    # Prediction error
    base_model_error = y_test - base_y_test_prediction

    # Mean squared error
    MSE = np.mean(base_model_error ** 2)
    # Root mean squared error
    RMSE = np.sqrt(MSE)
    # Mean absolute error
    MAE = np.mean(abs(base_model_error))
    # R squared
    R2 = 1 - sum(base_model_error ** 2) / sum((y_test - np.mean(y_test)) ** 2)
    # Mean absolute percentage error
    MAPE = np.mean(abs(base_model_error / y_test))

    print('************ BASELINE MODEL ************')
    print(f'The MSE for the model is {MSE:.2f}')
    print(f'The RMSE for the model is {RMSE:.2f}.')
    print(f'The MAE for the model is {MAE:.2f}.')
    print(f'The R-squared for the model is {R2:.2f}.')
    print(f'The MAPE for the model is {MAPE:.2f}.')






    # Appending categorical embeddings to the training dataset
    X_train_emb = pd.merge(X_train, cat_emb_df, left_on='Week Day', right_on='cat', how='inner').drop(['Week Day', 'cat', 'cat_index'], axis=1)

    # Appending categorical embeddings to the testing dataset
    X_test_emb = pd.merge(X_test, cat_emb_df, left_on='Week Day', right_on='cat', how='inner').drop(['Week Day', 'cat', 'cat_index'], axis=1)

    print(X_train_emb.info())
    print(X_test_emb.info())


    # Initiating the RF model
    emb_rf = RandomForestRegressor()
    print(y_train)
    emb_rf.fit(X_train_emb, y_train)
    emb_y_test_prediction = emb_rf.predict(X_test_emb)

    #g = plt.plot(y_test - emb_y_test_prediction, marker='o', linestyle='')
    #plt.show()

    # Model error
    emb_model_error = y_test - emb_y_test_prediction


    MSE = np.mean(emb_model_error**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(abs(emb_model_error))
    R2 = 1 - sum(emb_model_error**2)/sum((y_test-np.mean(y_test))**2)
    MAPE = np.mean(abs(emb_model_error/y_test))

    print(f'The MSE for the model is {MSE:.2f}')
    print(f'The RMSE for the model is {RMSE:.2f}.')
    print(f'The MAE for the model is {MAE:.2f}.')
    print(f'The R-squared for the model is {R2:.2f}.')
    print(f'The MAPE for the model is {MAPE:.2f}.')











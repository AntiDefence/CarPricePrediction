# CarPricePrediction

Introduction:
Welcome to the repository containing the code for predicting car prices using neural networks. This repository demonstrates how to preprocess car price data, build a neural network model using Keras, and optimize its hyperparameters using grid search. The goal is to predict car prices based on various features such as engine volume, mileage, and more.

Code Explanation:

Data Preprocessing:

The code begins by importing necessary libraries: pandas, numpy, LabelEncoder from sklearn.preprocessing, and modules from keras.
The car price prediction dataset is read from a RAR file and loaded into a pandas DataFrame called veri.
The ID column is dropped from the DataFrame.
Missing values in the Levy column are handled by replacing hyphens with NaN and then filling NaN values with 0. The column is converted to integer type.
The Turbo column is derived from the Engine volume column to indicate whether a car has a turbocharged engine or not.
The Engine volume and Mileage columns are processed by removing units and converting them to appropriate numeric types.
The Doors column is cleaned and converted to integer type.
Categorical columns are encoded using LabelEncoder.

Data Splitting and Scaling:

The target variable y (car prices) and the feature matrix x are defined.
Features are standardized using StandardScaler.
The dataset is split into training, validation, and test sets using train_test_split.

Neural Network Model:

A function modelkur is defined to create a sequential neural network model with customizable hyperparameters. #modelkurt is Turkish, and its meaning is defining model :)
The function takes parameters like units, activation, learning_rate, hidden_layers, and dropout_rate to configure the architecture of the neural network.
The model is compiled with the Adam optimizer, mean absolute error (MAE) loss, and MAE metric.

Hyperparameter Optimization:

A parameter grid parametreler is defined to specify hyperparameters for the grid search.
A KerasRegressor is created using the defined model function.
Grid search (GridSearchCV) is performed to find the best hyperparameters using cross-validation.
The best parameters are extracted from the grid search results.

Model Training and Prediction:

The best model is instantiated using the best hyperparameters.
The model is trained on the training data and validated using the validation data.
Predictions are made on the test set.

Results Visualization:

Matplotlib is used to create visualizations.
Two subplots are created: one for training and validation loss over epochs, and another for comparing actual vs. predicted car prices for a subset of test data.

Instructions:

Ensure you have the necessary libraries installed (numpy, pandas, sklearn, keras, matplotlib).
Extract the dataset from the RAR file and adjust the file path accordingly.
Run the code in your preferred Python environment.
The code will preprocess the data, build and train the neural network, and visualize the results.

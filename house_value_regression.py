import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler


class Regressor(BaseEstimator):

    def __init__(self, nb_epoch=1000, minibatch_size=10, learning_rate=0.01):
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        self.nb_epoch = nb_epoch
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.model = None
        self.binarizer = None
        self.scaler = None
        # Neural Network initialize in fit()

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
              The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        x = x.fillna(0)

        # Saves binarizer and scaler instances
        if training:
            # Initialize
            self.binarizer = LabelBinarizer()
            self.scaler = MinMaxScaler()

            one_hot_vectors = self.binarizer.fit_transform(x['ocean_proximity'])
            x = pd.concat([x, pd.DataFrame(one_hot_vectors, columns=self.binarizer.classes_)], axis=1)
            x = x.drop(['ocean_proximity'], axis=1)
            # Scale the features
            x = pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns)
        else:
            one_hot_vectors = self.binarizer.transform(x['ocean_proximity'])
            x = pd.concat([x, pd.DataFrame(one_hot_vectors, columns=self.binarizer.classes_)], axis=1)
            x = x.drop(['ocean_proximity'], axis=1)
            x = pd.DataFrame(self.scaler.transform(x), columns=x.columns)

        return x, y

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        # Error check
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y is not pandas DataFrame")

        X, Y = self._preprocessor(x, y=y, training=True)
        # Neural Network
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Convert
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.nb_epoch):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        # Preprocess the data and convert
        X, _ = self._preprocessor(x, training=False)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            prediction = self.model(X_tensor)

        return prediction.numpy()

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        predictions = self.predict(x)
        Y_numpy = y.values

        # Calculate mean squared error
        mean_sqd_err = np.mean((predictions - Y_numpy) ** 2)
        return mean_sqd_err


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """

    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """

    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def regressor_hyper_parameter_search(x_train, y_train):
    """
    Performs a hyperparameter for fine-tuning the regressor implemented
    in the Regressor class.
        
    Returns:
        The function should return your optimised hyperparameters.

    """

    param_grid = {
        'nb_epoch': [100, 200, 500, 1000, 2000],
        'minibatch_size': [8, 16, 32, 64, 128],
        'learning_rate': [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    }

    hyp_searcher = GridSearchCV(Regressor(), param_grid, cv=5)
    hyp_searcher.fit(x_train, y_train)

    return hyp_searcher.best_params_


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    regressor = Regressor(nb_epoch=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    test_hyperparameters = True
    if test_hyperparameters:
        # Hyper-parameter fine-tuning
        best_params = regressor_hyper_parameter_search(x_train, y_train)
        print("Best parameters: {}".format(best_params))
        best_regressor = Regressor(x_train, **best_params)
        best_regressor.fit(x_train, y_train)
        save_regressor(best_regressor)
        # Error
        error = regressor.score(x_train, y_train)
        print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

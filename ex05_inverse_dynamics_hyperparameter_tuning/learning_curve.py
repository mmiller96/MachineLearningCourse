import numpy as np
from sarcos import download_sarcos
from sarcos import load_sarcos
from sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD
import sklearn
import matplotlib.pyplot as plt
import time

# plot mean and variance of validation and training error for different training sizes
def plot_learning_curve(train_sizes, train_scores, valid_scores, path=None):
    train_scores_mean, valid_scores_mean = -train_scores.mean(axis=1), -valid_scores.mean(axis=1)
    train_scores_std, valid_scores_std = train_scores.std(axis=1), valid_scores.std(axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error', color='r')
    plt.plot(train_sizes, valid_scores_mean, 'o-', label='Validation error', color='g')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='g')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training size', fontsize=14)
    plt.legend()
    if(path is not None):
        plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    download_sarcos()
    X_train, Y_train = load_sarcos('train')
    X_test, Y_test = load_sarcos('test')

    target_scaler, feature_scaler = StandardScaler(), StandardScaler()
    Y_train, X_train = target_scaler.fit_transform(Y_train), feature_scaler.fit_transform(X_train)
    F = Y_train.shape[1]
    D = (X_train.shape[1],)
    layers = \
    [
        {
            "type": "fully_connected",
            "num_nodes": 50
        }
    ]
    model = MultilayerNeuralNetwork(D, F, layers, training='regression', std_dev=0.01, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, batch_size=32, alpha=0.005, eta=0.5,  verbose=2)

    time_start = time.time()
    train_sizes, train_scores, valid_scores = learning_curve(mbsgd, X_train, Y_train, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 0.9, 20), cv=5, n_jobs=-1)
    time_end = time.time()
    print('Minutes: {:10.2f}'.format((time_end - time_start) / 60))
    plot_learning_curve(train_sizes, train_scores, valid_scores, path=r'C:\Users\Data Miner\Desktop\Uni\MachineLearning\ex05\inverse_dynamics\Bilder\learning_curve_training_sizes.pdf')

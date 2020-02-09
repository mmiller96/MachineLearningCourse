from sarcos import download_sarcos
from sarcos import load_sarcos
from sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD
import time
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

def save_error_params(train_error_list, test_error_list, param_list, path):
    indices = np.argsort(np.argsort(test_error_list))
    sorted_param = [param for _, param in sorted(zip(indices, param_list))]
    sorted_train_error = [error for _, error in sorted(zip(indices, train_error_list))]
    sorted_test_error = [error for _, error in sorted(zip(indices, test_error_list))]
    data = {'params': sorted_param, 'train_error': sorted_train_error, 'test_error': sorted_test_error}
    df = pd.DataFrame(data=data)
    df.to_csv(path, index=False)
    
if __name__ == "__main__":
    download_sarcos()
    X_train, Y_train = load_sarcos('train')
    X_train, Y_train = X_train[0:30000], Y_train[0:30000]
    target_scaler, feature_scaler = StandardScaler(), StandardScaler()
    Y_train, X_train = target_scaler.fit_transform(Y_train), feature_scaler.fit_transform(X_train)
    layers = [{"type": "fully_connected", "num_nodes": 90}]
    parameters = {'alpha': [0.0005, 0.003, 0.01], 'alpha_decay': [0.95, 0.97, 1], 'batch_size': [30, 55, 80],  'eta': [0.2, 0.5, 0.8], 'eta_inc': [0, 0.00001]}
    scores, params = [], []
    F = Y_train.shape[1]
    D = (X_train.shape[1],)
    model = MultilayerNeuralNetwork(D, F, layers, training='regression', std_dev=0.01, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=50, verbose=0)

    time_start = time.time()
    clf = GridSearchCV(mbsgd, parameters, cv=3, n_jobs=3)
    clf.fit(X_train, Y_train)
    time_end = time.time()
    print('Minutes: {:10.2f}'.format((time_end - time_start) / 60))
    
    params = clf.cv_results_['params']
    test_error = clf.cv_results_['mean_test_score']
    train_error = clf.cv_results_['mean_train_score']
    save_error_params(train_error, test_error, params, r'C:\Users\Markus Miller\Desktop\Uni\Machine Learning\ex05\inverse_dynamics\grid_search.csv')
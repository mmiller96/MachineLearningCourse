import numpy as np
from sarcos import download_sarcos
from sarcos import load_sarcos
from sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD
import sklearn
import matplotlib.pyplot as plt
import time
import pandas as pd

def calc_error_different_layers(layer1_num, layer2_num=None, layer3_num=None):
    if(layer2_num is None):
        layers = [{"type": "fully_connected", "num_nodes": nodes1}]
        param = str(nodes1)
    elif(layer3_num is None):
        layers = [{"type": "fully_connected", "num_nodes": nodes1}, {"type": "fully_connected", "num_nodes": nodes2}]
        param = str(nodes1) + '_' + str(nodes2)
    else:
        layers = [{"type": "fully_connected", "num_nodes": nodes1}, {"type": "fully_connected", "num_nodes": nodes2}, {"type": "fully_connected", "num_nodes": nodes3}]
        param = str(nodes1) + '_' + str(nodes2) + '_' + str(nodes3)

    model = MultilayerNeuralNetwork(D, F, layers, training='regression', std_dev=0.01, verbose=True)
    mbsgd = MiniBatchSGD(net=model, batch_size=80, alpha=0.005, alpha_decay=0.99, epochs=50, verbose=0)
    cv_results = cross_validate(mbsgd, X_train, Y_train, cv=3, n_jobs=3)
    test_error = cv_results['test_score'].mean()
    train_error = cv_results['train_score'].mean()
    return train_error, test_error, param

def save_error_params(train_error_list, test_error_list, param_list, path):
    indices = np.argsort(np.argsort(test_error_list))
    sorted_param = [param for _, param in sorted(zip(indices, param_list))]
    sorted_train_error = [error for _, error in sorted(zip(indices, train_error_list))]
    sorted_test_error = [error for _, error in sorted(zip(indices, test_error_list))]
    data = {'params': sorted_param, 'train_error': sorted_train_error, 'test_error': sorted_test_error}
    df = pd.DataFrame(data=data)
    df.to_csv(path, index=False)

download_sarcos()
X_train, Y_train = load_sarcos('train')
X_train, X_test, Y_train, Y_test = X_train[0:30000], Y_train[0:30000]
target_scaler, feature_scaler = StandardScaler(), StandardScaler()
Y_train, X_train = target_scaler.fit_transform(Y_train), feature_scaler.fit_transform(X_train)
F = Y_train.shape[1]
D = (X_train.shape[1],)
nodes_layer1 = np.array([65, 67, 70, 72, 75, 77, 80, 82, 85, 87, 90, 92, 95, 97, 100])
train_error_list, test_error_list, param_list = [], [], []

time_start = time.time()
for nodes1 in nodes_layer1:
    train_error, test_error, param = calc_error_different_layers(nodes1, layer2_num=None, layer3_num=None)
    train_error_list.append(train_error)
    test_error_list.append(test_error)
    param_list.append(param)
time_end = time.time()
print('Minutes: {:10.2f}'.format((time_end - time_start) / 60))
save_error_params(train_error_list, test_error_list, param_list, r'C:\Users\Markus Miller\Desktop\Uni\Machine Learning\ex05\inverse_dynamics\search_architecture_error2.csv')
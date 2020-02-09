from sarcos import download_sarcos
from sarcos import load_sarcos
from sarcos import nMSE
from sklearn.preprocessing import StandardScaler
from multilayer_neural_network import MultilayerNeuralNetwork
from minibatch_sgd import MiniBatchSGD
import time
import numpy as np

if __name__ == "__main__":
    download_sarcos()
    X_train, Y_train = load_sarcos('train')
    X_test, Y_test = load_sarcos('test')
    
    target_scaler, feature_scaler = StandardScaler(), StandardScaler()
    Y_train, X_train = target_scaler.fit_transform(Y_train), feature_scaler.fit_transform(X_train)
    Y_test, X_test = target_scaler.transform(Y_test), feature_scaler.transform(X_test) 
    
    layers = [{"type": "fully_connected", "num_nodes": 90}]    
    F = Y_train.shape[1]
    D = (X_train.shape[1],)
    model = MultilayerNeuralNetwork(D, F, layers, training='regression', std_dev=0.01, verbose=True)
    mbsgd = MiniBatchSGD(net=model, epochs=100, alpha=0.003, alpha_decay=1, batch_size=80, eta=0.5, eta_inc=0, verbose=2)
    mbsgd.fit(X_train, Y_train)
    
    Y_pred_train=model.predict(X_train)          # Predict Y from training set
    print("Train set:")
    MnMSE=100 * nMSE(Y_pred_train, Y_train)
    print("nMSE =" , MnMSE, "%")
    for f in range(F):                  # Print nMSE for different dimensions
        print("Dimension %d: nMSE = %.2f %%" % (f + 1, 100 * nMSE(Y_pred_train[:, f], Y_train[:, f])))
    
    print("")
    Y_pred_test=model.predict(X_test)          # Predict Y from test set
    print("Test set:")
    MnMSE=100 * nMSE(Y_pred_test, Y_test)
    print("nMSE =" , MnMSE, "%")
    for f in range(F):                  # Print nMSE for different dimensions
        print("Dimension %d: nMSE = %.2f %%" % (f + 1, 100 * nMSE(Y_pred_test[:, f], Y_test[:, f])))
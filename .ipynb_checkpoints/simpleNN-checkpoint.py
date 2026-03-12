import numpy as np

# Defining activation functions and their derivatives for readability later on in the code
def ReLU(v):
    return np.maximum(0, v)

def deriv_ReLU(v):
    out = []
    for i in v:
        out.append(1) if i > 0 else out.append(0)
    return np.array(out)

def sigmoid(v):
    return 1 /(1 + np.exp(-v))

# Forward-propagation
def forwardprop(inputv, A, a):
    """"
    ReLU(Ai @ v + ai), where v is the input vector
    args:
        inputv (array) the input vector that has length = number of features
        A (list) a list of weight matrices, len(A) = # of hidden layers
        a (list) a list of bias vectors/arrays, len(a) = # of hidden layers
    returns:
        (array) the output vector to push into the output layer
    """
    assert(len(A) == len(a))
    nlayers = len(A)
    z = inputv
    preacts = []
    activations = [inputv]

    for i in range(nlayers - 1): # hidden layers
        u = A[i] @ z + a[i]
        preacts.append(u)
        z = ReLU(u)
        activations.append(z)
    
    u_out = A[-1] @ z + a[-1]
    z_out = sigmoid(u_out)
    preacts.append(u_out)
    activations.append(z_out)
    return preacts, activations, z_out

# Classification
def classify(z_out, threshold):
    classify_vector = []
    
    for i in z_out:
        classify_vector.append(True) if i > threshold else classify_vector.append(False)
    
    return classify_vector

# Calculating loss
def bce_loss(y, y_hat):
    # Adding a very small number just in case ln(y_hat) or ln(1- y_hat) blows up
    return -np.mean(y * np.log(y_hat + 1e-8) - (1 - y) * np.log(1 - y_hat + 1e-8))

# Backwardspropagation
def backwardprop(y, y_hat, preacts, activations, A, a, eta):
    """
    Update the weights of the matrices and bias vectors inside list A, a
     args:
        y (array) true label (0 or 1)
        y_hat (array) predicted probability ŷ
        preacts (list) pre-activation values z from forwardprop
        activations (list) activation values from forwardprop
        A (list) weight matrices (same as forwardprop)
    """
    nlayers = len(A)
    # Conceptually sig_deriv = y_hat[0] * (1 - y_hat[0]), dL_dyhat = (y_hat - y) / sig_deriv
    # delta = dL_dyhat * sig_deriv
    delta = (y_hat - y) 
    A[-1] -= eta * np.outer(delta, activations[-2])
    a[-1] -= eta * delta

    for i in range(nlayers - 2, -1, -1):
        delta_z = A[i + 1].T @ np.atleast_1d(delta)
        delta = delta_z * deriv_ReLU(preacts[i])
        A[i] -= eta * np.outer(delta, activations[i])
        a[i] -= eta * delta
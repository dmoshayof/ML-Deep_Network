import numpy as np

STUDENT = {'name': 'Ariel Vetzler_Daniel Moshayof',
           'ID': '207458688_311126668'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.

    # My code:
    c = np.max(x)
    ex = np.exp(x - c)
    x = ex / np.sum(ex)
    # End of code.

    return x


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    W, b = params

    # YOUR CODE HERE.
    z = np.dot(x, W) + b
    probs = softmax(z)
    # End Of Code.

    return probs


def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W, b = params
    # YOU CODE HERE

    # outputLayer = softmax ( wx + b )
    pred_vec = classifier_output(x, params)
    y_hat = pred_vec[y]
    # loss = - log( softmax( y_hat ) )
    loss = - np.log(y_hat)
    '''
    dl\dw = dl\dz * dz\dw
    dl\db = dl\dz * dz\dw
    dl\dz = y_hat - y
    dz\dw = x
    dz\db = 1

    dl\dw = gw = (y_hat-y)*x
    dl\db = gb = (y_hat-y)*1  = y_hat - y
    '''

    # One Hot Vector - 1 in the right index.
    y_hot_vector = np.zeros(pred_vec.shape)
    y_hot_vector[y] = 1

    gb = pred_vec - y_hot_vector
    gW = np.outer(x, gb)

    return loss, [gW, gb]


def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W, b]


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1, 2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001, 1002]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001, -1002]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b = create_classifier(3, 4)


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)

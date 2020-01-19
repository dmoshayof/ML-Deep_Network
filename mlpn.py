import numpy as np

STUDENT = {'name': 'Ariel Vetzler_Daniel Moshayof',
           'ID': '207458688_311126668'}

from loglinear import softmax


def classifier_output(x, params):
    # YOUR CODE HERE.
    # z is the layer before activate the activation function.
    z_layers = []
    # h is the layer after activation function.
    h_layers = []

    h = x
    for index in range(0, len(params), 2):
        w = params[index]
        b = params[index + 1]
        z = np.dot(h, w)
        z = np.add(z, b)
        h = np.tanh(z)
        z_layers.append(z)
        h_layers.append(h)

    h_layers.pop()
    z_layers.pop()
    probs = softmax(z)
    return probs, z_layers, h_layers


def predict(x, params):
    predictionVec, _, _ = classifier_output(x, params)
    return np.argmax(predictionVec)


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    pred_vec, z_layers, h_layers = classifier_output(x, params)

    y_hat = pred_vec[y]
    loss = - np.log(y_hat)

    y_hot_vector = np.zeros(pred_vec.shape)
    y_hot_vector[y] = 1
    gb = pred_vec - y_hot_vector
    gWs_gbs = []
    gWs_gbs.insert(0, gb)
    Ws = params[0::2]

    for index in range(len(Ws) - 1):
        # print(index)
        '''dloss\dw'''
        # dl\dw = dl\dz * dz\dw
        dz_dW = h_layers[-(index + 1)].T
        gW = np.outer(dz_dW, gb)
        gWs_gbs.insert(0, gW)

        # dz_dh = 1 - np.tanh(z_layers[-(index + 1)] ** 2)
        U = Ws[-(index + 1)]
        dz_dh = 1 - (np.tanh(z_layers[-(index + 1)]) ** 2)
        dz_dh = U.T * dz_dh
        gb = np.dot(gb, dz_dh)
        gWs_gbs.insert(0, gb)

    gFirst_w = np.outer(x, gb)
    gWs_gbs.insert(0, gFirst_w)
    return loss, gWs_gbs


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    # Xavier Glorot et al's suggestion:
    for dim1, dim2 in zip(dims, dims[1:]):
        # sqrt 6  / sqrt m+n
        epsilon = np.sqrt(6) / (np.sqrt(dim1 + dim2))
        params.append(np.random.uniform(-epsilon, epsilon, [dim1, dim2]))
        epsilon = np.sqrt(6) / (np.sqrt(dim2))
        params.append(np.random.uniform(-epsilon, epsilon, dim2))
    #
    # for first_dim, second_dim in zip(dims, dims[1:]):
    #     W = np.zeros((first_dim, second_dim))
    #     b = np.zeros(second_dim)
    #     # Randomize the values so the gradients will change.
    #     W = np.random.randn(W.shape[0], W.shape[1])
    #     b = np.random.randn(b.shape[0])
    #     params.append(W)
    #     params.append(b)

    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag, V, b_tag_tag = create_classifier([2, 2, 2, 2])


    def _loss_and_W_grad(W):
        global b, U, b_tag, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W, U, b_tag, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W, b, b_tag, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W, U, b, V, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[3]


    def _loss_and_V_grad(V):
        global W, U, b, b_tag, b_tag_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[4]


    def _loss_and_b_tag_tag_grad(b_tag_tag):
        global W, U, b, V, b_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag, V, b_tag_tag])
        return loss, grads[5]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        U = np.random.randn(U.shape[0], U.shape[1])
        V = np.random.randn(V.shape[0], V.shape[1])
        b = np.random.randn(b.shape[0])
        b_tag = np.random.randn(b_tag.shape[0])
        b_tag_tag = np.random.randn(b_tag_tag.shape[0])

        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_V_grad, V)
        gradient_check(_loss_and_b_tag_tag_grad, b_tag_tag)

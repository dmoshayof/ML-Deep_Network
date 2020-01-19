import numpy as np

STUDENT = {'name': 'Ariel Vetzler_Daniel Moshayof',
           'ID': '207458688_311126668'}
from loglinear import softmax


def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag = params
    # f(x) = softmax(Utanh(Wx+b)+b')
    h = np.tanh(np.dot(x, W) + b)
    probs = softmax(np.dot(h, U) + b_tag)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W, b, U, b_tag = params

    pred_vec = classifier_output(x, params)
    y_hat = pred_vec[y]
    # loss = - log(softmax(y_hat))
    loss = - np.log(y_hat)
    # One Hot Vector - 1 in the right index.
    y_hot_vector = np.zeros(pred_vec.shape)
    y_hot_vector[y] = 1

    ################################

    # z = Utanh(Wx+b) + b'
    # h = Wx+b
    #
    # dloss\dz  = y_hat - y
    # dz\dh = U(1-tan(h)^2)
    # dh\dw = x
    # dz\dU = tanh(Wx+b)
    # dh\db = 1
    # dz\db' = 1
    # dloss\dw = dl\dz * dz\dh * dh\dw
    # dloss\db = dl\dz * dz\dh * dh\db
    # dloss\dU = dl\dz * dz\dU
    # dloss\db' = dl\dz * dz\db'

    ################################

    # y_hat - y
    dl_dz = pred_vec - y_hot_vector
    # Inside function = Wx + b
    h = np.dot(x, W) + b
    # np.tanh(np.dot(x, W) + b)
    dh_dw = x

    ################################

    '''dloss\db_tag'''
    dz_db_tag = 1
    # (prediction_Vec - y_Vec) * 1
    gb_tag = pred_vec - y_hot_vector

    '''dloss\dU'''
    # h = np.dot(x, W) + b
    dz_dU = np.tanh(h)
    # TODO
    gU = np.outer(gb_tag, dz_dU).T

    '''dloss\db'''
    ''' dz_dh = U * [1 - (tanh(Wx+b)^2)]'''
    dz_dh = 1 - np.tanh(np.dot(x, W) + b) ** 2
    # TODO
    dz_dh = U.T * dz_dh
    # gb = dl_dz * dz_dh
    gb = np.dot(gb_tag, dz_dh)

    '''dloss\dw'''
    ''' (y_hat - y) * [ U * [1 - (tanh(Wx+b)^2)] ] * x '''
    gW = np.outer(x, gb)

    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.ones((in_dim, hid_dim))
    b = np.ones(hid_dim)
    U = np.ones((hid_dim, out_dim))
    b_tag = np.ones(out_dim)

    W = np.random.randn(W.shape[0], W.shape[1])
    U = np.random.randn(U.shape[0], U.shape[1])
    b = np.random.randn(b.shape[0])
    b_tag = np.random.randn(b_tag.shape[0])
    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(2, 2, 2)


    def _loss_and_W_grad(W):
        global b, U, b_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W, U, b_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W, b, b_tag
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W, U, b
        loss, grads = loss_and_gradients([1, 2], 0, [W, b, U, b_tag])
        return loss, grads[3]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        U = np.random.randn(U.shape[0], U.shape[1])
        b = np.random.randn(b.shape[0])
        b_tag = np.random.randn(b_tag.shape[0])

        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)

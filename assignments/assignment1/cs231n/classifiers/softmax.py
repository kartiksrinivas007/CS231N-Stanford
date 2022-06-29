from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    f  = X @ W
    for i in range (X.shape[0]):
      #for every example in the minibatch 
        f = X[i,:] @ W;
        f -= np.max(f);
        exp_stuff = np.exp(f)/(np.sum(np.exp(f)));
        loss += -np.log(exp_stuff[y[i]]);
        for j in range (W.shape[1]):
            dW[:,j] += X[i] *exp_stuff[j];
        dW[:,y[i]] -= X[i];
    pass

    loss /= X.shape[0];
    dW /= X.shape[0];

    loss += reg*np.sum(W*W);
    dW += 2*reg*W;

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    nt = X.shape[0]
    scores  = X @ W;
    scores = scores - np.max(scores,axis = 1,keepdims = True); # to do a particular operation row wise 
    sum_exp_stuff = np.sum(np.exp(scores),axis = 1,keepdims = True); #make sure sizes are good for elementwise operations
    probab_matrix = np.exp(scores)/(sum_exp_stuff)
    loss += np.sum(-np.log(probab_matrix[np.arange(nt),y])) # fnatastic , how to generate the indices of an array

    probab_matrix[np.arange(nt),y] -= 1
    dW = X.T.dot(probab_matrix)

    loss /= nt;
    dW /= nt;

    loss = loss + reg*(np.sum(W*W));
    dW = dW + 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    用嵌套循环的方式实现朴素版的Softmax损失函数

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
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        f_i = X[i].dot(W)
        #计算i在每个分类的得分
        f_i -= np.max(f_i)
        #为了避免数值不稳定，每个分值向量都减去向量中的最大值
        sum_j = np.sum(np.exp(f_i))
        p = lambda k:np.exp(f_i[k]) / sum_j
        loss += -np.log(p(y[i]))
        
        #计算梯度：
        for k in range(num_classes):
            p_k = p(k)
            dW[:,k] += (p_k - (k == y[i])) * X[i]
    loss /= num_train #求完每个x的每个类别上的loss之后的和再除以总x数量作为Loss的平均值
    loss += 0.5 * reg * np.sum(W*W) #这一项是正则化项
    dW /= num_train
    dW += reg * W
        

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

    num_train = X.shape[0]
    f = X.dot(W)
    f -= np.max(f,axis = 1,keepdims = True)
    #如果不Keepdims就会报错，要Keepdims才能进行广播
    sum_f = np.sum(np.exp(f),axis =1,keepdims = True)
    p = np.exp(f)/sum_f
    
    loss= np.sum(-np.log(p[np.arange(num_train),y]))
    
    ind = np.zeros_like(p)
    ind[np.arange(num_train),y] = 1
    dW = X.T.dot(p - ind)
    
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train
    dW += reg * W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

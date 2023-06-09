from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
                #这一步保证了正确分类的时候不计算margin
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] += -X[i,:].T
                dW[:,j] += X[i,:].T

     # 现在损失函数是所有训练样本的和．但是我们要的是它们的均值．所以我们用　num_train 来除．
    loss /= num_train
    dW /= num_train

    # 加入正则项
    loss += reg * np.sum(W * W)
    dW += reg * W



    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    
    scores_correct = scores[np.arange(num_train),y]
    # 对于第i个样本，正确类别的分数可以通过scores[i, y[i]]来获取，这与scores_correct[i]的值相同。
    scores_correct = np.reshape(scores_correct,(num_train,-1))
    margins = scores - scores_correct +1
    margins = np.maximum(0,margins)
    #将margins矩阵中小于0的元素置为0，得到修正后的margins矩阵。
    margins[np.arange(num_train),y] = 0
    #将margins矩阵中每个样本的正确类别的分数与差值设置为0（原来是△，本例子的△为1）
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W*W)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1
    row_sum = np.sum(margins,axis=1)
    margins[np.arange(num_train),y] = -row_sum
    #达到什么目的？：j≠yi且scores - scores_correct +1＜0时，导数（margin）=0，这一步在上面的np.maximum实现；j≠yi且scores - scores_correct +1＞0时，导数（margin）=1；j=yi时，导数＝-[（margins＞0）的个数]
    dW += np.dot(X.T,margins) / num_train + reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from ..classifiers.linear_svm import *
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        
        使用随机梯度下降来训练这个分类器。
        输入:
        -X:一个 numpy array，形状为 (N, D), 存储训练数据。 共N个训练数据， 每个训练数据是N维的。
        -Y:一个 numpy array， 形状为(N,), 存储训练数据的标签。y[i]=c 表示 x[i]的标签为ｃ，其中 0 <= c <= C 。
        -learning rate： float， 优化的学习率。
        -reg：float， 正则化强度。
        -num_iters: integer， 优化时训练的步数。
        -batch_size: integer, 每一步使用的训练样本数。
        -verbose: boolean，若为真，优化时打印过程。
        输出：
        一个存储每次训练的损失函数值的list

        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
              # 任务: 
              # 从训练集中采样batch_size个样本和对应的标签，在这一轮梯度下降中使用。
              # 把数据存储在 X_batch 中，把对应的标签存储在 y_batch 中。
              # 采样后，X_batch 的形状为 (batch_size,dim),y_batch的形状为(batch_size,)
              #  
              # 提示：用 np.random.choice 来生成 indices。有放回采样的速度比无放回采
              # 样的速度要快 ps即可重复选择

            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            batch_inx = np.random.choice(num_train,batch_size)
            X_batch = X[batch_inx,:]
            y_batch = y[batch_inx]
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # 使用梯度和学习率更新权重          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W = self.W - learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores = X.dot(self.W)
        y_pred = np.argmax(scores,axis=1)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

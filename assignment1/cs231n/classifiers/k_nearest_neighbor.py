from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        通过一个两层的嵌套循环，遍历测试样本点，并求其到全部训练样本点的距离

        输入:
        - X: 测试数据集，一个 (num_test, D) 大小的numpy数组

        返回:
        - dists: 一个 (num_test, num_train) 大小的numpy数组，其中dists[i, j]表示测试样本i到训练样本j的欧式距离

         """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j]= np.sqrt(np.sum(np.square(self.X_train[j]-X[i])))
                
        #####################################################################
        # 任务:                                                            
        # 计算第i个测试点到第j个训练样本点的L2距离，并保存到dists[i, j]中
        # 注意不要在维度上使用for循环                                       
        #####################################################################

        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis = 1))

            pass

            # X_train的所有行到X_test的第i行的距离，是二维的。上面np.square(self.X_train[j]-X[i])是X_train的第j行到X_test的第i行的距离，是一维的。所以这里有axis=1而上面没有。
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.multiply(np.dot(X,self.X_train.T),-2) 
        sq1 = np.sum(np.square(X),axis=1,keepdims = True) 
#sq1在不keepdims的时候是行向量的数组（一维，500个数），无法与dists的每行5000个数匹配，Keepdim以后它就变成了列向量（二维，500个数）（500,1），可以与dists的列相加了（广播）。而sq2在不keepdim的时候是行向量的数组（一维，5000个数），可以与dists的每行5000个数匹配，因此它不需要Keepdim就可以直接广播
        sq2 = np.sum(np.square(self.X_train),axis=1) 
        dists = np.add(dists,sq1) 
        dists = np.add(dists,sq2) 
        dists = np.sqrt(dists)


        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k):
        """
        通过距离矩阵，预测每一个测试样本的类别

        输入：
        - dists: 一个(num_test, num_train) 大小的numpy数组，其中dists[i, j]表示第i个测试样本到第j个训练样本的距离

        返回：
        - y: 一个 (num_test,)大小的numpy数组，其中y[i]表示测试样本X[i]的预测结果
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
           # 一个长度为k的list数组，其中保存着第i个测试样本的k个最近邻的类别标签
            closest_y = []
            #########################################################################
      # 任务:                                                                 
      # 通过距离矩阵找到第i个测试样本的k个最近邻,然后在self.y_train中找到这些 #
      # 最近邻对应的类别标签，并将这些类别标签保存到closest_y中。             
      # 提示: 可以尝试使用numpy.argsort方法                                 
      #########################################################################


            closest_y = self.y_train[np.argsort(dists[i])[:k]]

            #########################################################################
      # 任务:                                                                 #
      # 现在你已经找到了k个最近邻对应的标签, 下面就需要找到其中出现最多的那个   #
      # 类别标签，然后保存到y_pred[i]中。如果有票数相同的类别，则选择编号小   #
      # 的类别                                                                #
      #########################################################################

            y_pred[i] = np.argmax(np.bincount(closest_y))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

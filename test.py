# from numpy.random import dirichlet

# a = dirichlet([10]*9, 2)

# print(a)

# for i in a:
#     print(sum(i))

# x = {"apple", "banana", "cherry"}
# y = {"google", "runoob", "apple"}
 
# z = x.union(y) 
 
# print(z)
# print(x)

# import numpy as np

# l = np.arange(5)
# print(l)

# import torch
# a  = torch.from_numpy(l)
# print(a.size())
# print(a.size() == torch.Size([]))
# print(a.size() == torch.Size([5]))

# import numpy as np
# from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
# from sklearn import svm

# x_train = np.loadtxt("/home/liang/DG-FL/dataset/UNSW-NB15-split2/train/s_train/x")
# y_train = np.loadtxt("/home/liang/DG-FL/dataset/UNSW-NB15-split2/train/s_train/y")

# x_valid = np.loadtxt("/home/liang/DG-FL/dataset/UNSW-NB15-split2/train/s_valid/x")
# y_valid = np.loadtxt("/home/liang/DG-FL/dataset/UNSW-NB15-split2/train/s_valid/y")

# x_test = np.loadtxt("/home/liang/DG-FL/dataset/UNSW-NB15-split2/train/t/x")
# y_test = np.loadtxt("/home/liang/DG-FL/dataset/UNSW-NB15-split2/train/t/y")

# print(x_train.shape)
# print(y_train.shape)

# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
# clf.fit(x_train, y_train.ravel())

# y_hat = clf.predict(x_valid)
# print(y_hat[:10])
# precision = precision_score(y_valid, y_hat)
# recall = recall_score(y_valid, y_hat)
# f1 = f1_score(y_valid, y_hat)

# y_hat = clf.predict(x_test)
# print(y_hat[:10])
# precision = precision_score(y_test, y_hat)
# recall = recall_score(y_test, y_hat)
# f1 = f1_score(y_test, y_hat)
# import numpy as np
# while 1:

#     lam = np.random.beta(0.2, 0.2)
#     print(lam)

# import numpy as np

# a = np.array([1,2,3,4,5,6])
# print(a==2)
# class test(object):
#     def __init__(self,a):
#         self.a = a 

# t = test(5)
# print(t.a)
# t.a = 7
# print(t.a)
# print(t)
a = [1,2,3,4,5]
a = [float(i)/15 for i in a]
print(a)


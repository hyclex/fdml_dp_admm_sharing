from sklearn import metrics
import numpy as np
import os
import scipy as sp
from sklearn.datasets import load_svmlight_file

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
	sigmoid_val = sigmoid(x)
	return sigmoid_val * (1 - sigmoid_val)

def logloss(true, pred):
	return metrics.log_loss(true, pred)

def logloss_logit_form(true, logit):
	# for +1, -1 label
	return np.mean(np.log(1+np.exp(-logit*true)))

def load_svmlightfile_data(train_path, test_path):
	tmp = load_svmlight_file(train_path)
	train_feature = tmp[0]
	train_label = tmp[1]
	tmp = load_svmlight_file(test_path)
	test_feature = tmp[0]
	test_label = tmp[1]
	# align the dimension of the train and test
	num_row_train, num_col_train = train_feature.shape
	num_row_test, num_col_test = test_feature.shape
	if num_col_train > num_col_test:
		test_feature = test_feature.toarray()
		test_feature = np.concatenate([test_feature, np.zeros([num_row_test, num_col_train - num_col_test])], axis=1)
	if num_col_train < num_col_test:
		test_feature = test_feature.toarray()
		test_feature = test_feature[:, 0:num_col_train]
	test_feature = sp.sparse.csr_matrix(test_feature, shape=[num_row_test, num_col_train])
	return train_feature, train_label, test_feature, test_label	

def csr_to_indices(X):
	coo = X.tocoo()
	return np.mat([coo.row, coo.col]).transpose()

def csr_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def load_raw(data_set = "a9a"):
	data_dir_path = "../data"
	result_dir_path = "../result"
	train_path = os.path.join(data_dir_path, data_set, "raw_train")
	test_path = os.path.join(data_dir_path, data_set, "raw_test")
	return load_svmlightfile_data(train_path, test_path)

def load_parts(data_set = "a9a", num_parts = 2):
	data_dir_path = "../data"
	result_dir_path = "../result"
	train_feature = []
	train_label = []
	test_feature = []
	test_label = []
	for i in range(num_parts):
		train_path = os.path.join(data_dir_path, data_set, "raw_train-part"+str(i))
		test_path = os.path.join(data_dir_path, data_set, "raw_test-part"+str(i))
		tmp_trf, tmp_trl, tmp_tef, tmp_tel = load_svmlightfile_data(train_path,test_path)
		train_feature.append(tmp_trf)
		train_label.append(tmp_trl)
		test_feature.append(tmp_tef)
		test_label.append(tmp_tel)
	return train_feature, train_label, test_feature, test_label



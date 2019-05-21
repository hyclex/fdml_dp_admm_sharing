# from scipy.optimize import minimize
from scipy import sparse
import numpy as np
import time
import tensorflow as tf

import util
from config import config

class lr:
	def __init__(self):
		self.x = None
		self.b = None
		self.feature_size = None
		self.train_sample_size = None
		self.lam = 1.0
		self.train_label = None
		self.train_feature = None

	def fit(self, train_feature, train_label):
		self.train_sample_size, self.feature_size = train_feature.shape
		self.x, self.b = self.fit_with_tf(train_feature, train_label)

	def predict_proba(self, test_feature):
		logit = test_feature.dot(self.x) + self.b
		return util.sigmoid(logit)


	def fit_with_tf(self, train_feature, train_label):
		my_label = train_label==1
		# define the place holders
		feature_input_indices = tf.placeholder(tf.int64, [None, 2])
		feature_input_values = tf.placeholder(tf.float32, [None])
		feature_input_shape = tf.placeholder(tf.int64, [2])

		feature_input = tf.SparseTensor(feature_input_indices, feature_input_values, feature_input_shape)
		label_holder = tf.placeholder(tf.float32, [None, 1])
		# construct tf model
		with tf.variable_scope("lr"+str(np.random.rand())):
			W = tf.get_variable("W", [self.feature_size, 1], initializer = tf.initializers.random_normal(0.1), trainable = True)
			B = tf.get_variable("B", initializer=tf.constant(0.), trainable = True)
			logit = tf.sparse_tensor_dense_matmul(feature_input, W) + B
			pred = tf.nn.sigmoid(logit)
			loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_holder, logits = logit)) + tf.contrib.layers.l2_regularizer(0.0000001)(W)
		opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
		# start training
		sess = tf.Session()
		batch_size = 100
		num_batch = self.train_sample_size / batch_size
		num_epoch = 40

		init = tf.global_variables_initializer()
		sess.run(init)
		# tf.get_variable_scope().reuse_variables()

		for epoch in range(num_epoch):
			avg_cost = 0
			for batch in range(num_batch):
				start_idx = batch * batch_size
				end_idx = (batch+1) * batch_size
				cur_cost, _ = sess.run([loss, opt], 
					feed_dict={
						feature_input_indices: util.csr_to_indices(train_feature[start_idx:end_idx]),
						feature_input_values: train_feature[start_idx:end_idx].data, 
						feature_input_shape: [batch_size, self.feature_size], 
						label_holder : my_label[start_idx:end_idx].reshape(batch_size, 1)})
				avg_cost += cur_cost / num_batch
			print("epoch %d: train loss: %6.4f" % (epoch, avg_cost))

		w, b = sess.run([W, B])
		# return the variables
		return w, b

	# def loss(self, w):
	# 	wt = w.reshape([-1, 1])
	# 	w[-1] = 0
	# 	logit = self.train_feature_aug.dot(wt)
	# 	return util.logloss_logit_form(self.train_label, logit) + self.lam * np.linalg.norm(w)

	# def loss_der(self, w):
	# 	wt = w.reshape([-1, 1])
	# 	wtx = w.reshape([-1, 1])
	# 	wtx[-1, 0] = 0
	# 	logit = self.train_feature_aug.dot(wt)
	# 	return (-(self.train_feature_aug).transpose().dot(util.sigmoid_der(logit * self.train_label.reshape([-1, 1]))* self.train_label.reshape([-1, 1])) + 2 * self.lam * wtx).flatten()


def test_lr(train_feature, train_label, test_feature, test_label):
	## build and train the model 
	lr_model = lr()
	time_start = time.time()
	lr_model.fit(train_feature, train_label)
	time_end = time.time()
	time_elapsed = time_end - time_start
	# # evaluate the result
	pred_proba_train = lr_model.predict_proba(train_feature)
	logloss_train = util.logloss(train_label, pred_proba_train)
	pred_proba_test = lr_model.predict_proba(test_feature)
	logloss_test = util.logloss(test_label, pred_proba_test)
	# print out
	print("Training logloss: %6.4f" % logloss_train)
	print("Training time: %6.2f" % time_elapsed)
	print("Testing logloss: %6.4f" % logloss_test)

if __name__ == "__main__":
	print("-----Evaluating a9a all...")
	train_feature, train_label, test_feature, test_label = util.load_a9a_raw()
	test_lr(train_feature, train_label, test_feature, test_label)

	print("-----Evaluating a9a part0...")
	train_feature, train_label, test_feature, test_label = util.load_a9a_parts()
	test_lr(train_feature[0], train_label[0], test_feature[0], test_label[0])

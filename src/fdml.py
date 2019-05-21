## TO DO
# data loading and initialization
# training
# recording and saving
# noise adding and handling

import os
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import time

from config import config
import util
reload(util)

class nodes:
	def __init__(self):
		pass
	def load_data(self, train_path, test_path):
		train_feature, train_label, test_feature, test_label = util.load_svmlightfile_data(train_path, test_path)
		self.train_feature = train_feature
		self.train_label = train_label
		self.test_feature = test_feature
		self.test_label = test_label
		self.size_feature = train_feature.shape[1]
		self.train_size = train_feature.shape[0]
		self.test_size = test_feature.shape[0]

class server(nodes):
	def __init__(self, config):
		self.config = config
		# loading data
		train_path = os.path.join(self.config["input_dir_path"], "raw_train-server")
		test_path = os.path.join(self.config["input_dir_path"], "raw_test-server")
		self.load_data(train_path, test_path)
		# initialize the parameters
		self.z = np.zeros(self.train_size)
		self.z_old = np.zeros(self.train_size)
		self.y = np.zeros(self.train_size)
		self.s_Q = np.zeros(self.train_size)

	def update(self, s_Q):
		self.s_Q = s_Q
		self.z_old = self.z[:]
		self.update_z()
		self.update_y()

	def update_z(self):
		res = minimize(self.h, self.z, method=self.config["z_solver"], jac=self.h_der, hess=self.h_hess, options={'disp': True})
		self.z = res.x

	def update_y(self):
		self.y = self.y + self.config["rho"] * (self.s_Q - self.z)

	def get_z_y(self):
		return self.z, self.y

	def h(self, z):
		l = self.l(z)
		linear = np.inner(self.y, z)
		aug = self.config["rho"] * 2 * np.linalg.norm(self.s_Q - z)**2
		return l - linear + aug

	def h_der(self, z):
		l_der = self.l_der(z)
		return l_der - self.y + self.config["rho"] * (z - self.s_Q)

	def h_hess(self, z):
		l_hess = self.l_hess(z)
		return l_hess + self.config["rho"]

	def l(self, z):
		zy = z*self.y
		return np.sum(np.log(1+np.exp(-zy)))

	def l_der(self, z):
		zy = z*self.y
		return -self.y / (1+np.exp(zy))

	def l_hess(self, z):
		zy = z*self.y
		ezy = np.exp(zy)
		return np.diag(self.y**2 * ezy / (1+ezy)**2)


class worker(nodes):
	def __init__(self, worker_id, config):
		self.worker_id = worker_id
		self.config = config
		# loading data
		train_path = os.path.join(self.config["input_dir_path"], "raw_train-part"+str(self.worker_id))
		test_path = os.path.join(self.config["input_dir_path"], "raw_test-part"+str(self.worker_id))
		self.load_data(train_path, test_path)
		# augenment for worker 0 to include a bias
		if 0 == worker_id:
			tmp = sparse.hstack([self.train_feature, np.ones([self.train_size,1])])
			self.train_feature = sparse.csr_matrix(tmp)
			tmp = sparse.hstack([self.test_feature, np.ones([self.test_size,1])])
			self.test_feature = sparse.csr_matrix(tmp)
			self.size_feature += 1

		# initialize the parameters
		self.x = np.zeros(self.size_feature)
		self.s_Q = np.zeros(self.train_size)
		# self.Qx_old = np.zeros(self.train_size)
		self.Qx = np.zeros(self.train_size)
		self.y = np.zeros(self.train_size)
		self.z = np.zeros(self.train_size)

	def update_no_noise(self, s_Q, z, y):
		self.s_Q = s_Q
		self.z = z
		self.y = y
		res = minimize(self.g, self.x, method=self.config["x_solver"], jac=self.g_der, hess=self.g_hess, options={'xtol': 1e-8, 'disp': True})
		self.x = res.x
		# self.Qx_old = self.Qx
		self.Qx = self.train_feature.dot(self.x.reshape([-1,1])).flatten()

	def update_with_added_noise(self):
		pass

	def get_x_norm(self):
		pass

	def get_Dx_no_noise(self, is_train=True):
		if True == is_train:
			return self.Qx
		else:
			return self.test_feature.dot(self.x.reshape([-1, 1])).flatten()

	def get_Dx_with_noise(self, is_train=True):
		pass

	def g(self, x):
		# x update optimization problem value function
		Dx = self.train_feature.dot(x)
		R = self.R(x)
		linear = np.inner(self.y, Dx)
		aug = self.config["rho"] / 2 * np.linalg.norm(self.s_Q - self.Qx + Dx - self.z)**2
		return self.config["lambda"] * R + linear + aug

	def g_der(self, x):
		# x update optimization problem derivative
		R_der = self.R_der(x)
		remain = self.train_feature.transpose().dot(self.y + self.config["rho"]*(self.s_Q - self.Qx + self.train_feature.dot(x)- self.z)).flatten()
		return self.config["lambda"] * R_der + remain

	def g_hess(self, x):
		# x update optimization problem hession matrix
		R_hess = self.R_hess(x)
		remain = self.config["rho"] * self.train_feature.transpose().dot(self.train_feature).toarray()
		return self.config["lambda"] + remain

	def R(self, x):
		return np.linalg.norm(x)

	def R_der(self, x):
		return 2*x

	def R_hess(self, x):
		return 2*np.eye(len(x))

class coord:
	def __init__(self, config):
		self.config = config
		if True == self.config["is_verbose"]:
			print("Initializing...")

		self._init_server()
		self._init_workder()

		self._set_model_loss()

		self._init_history()

	def _init_server(self):
		self.server = server(self.config)
	def _init_workder(self):
		self.worker = []
		for i in range(self.config["num_workers"]):
			self.worker.append(worker(i, self.config))
	def _set_model_loss(self):
		if self.config["model"] == "lr":
			self.loss = util.logloss
			self.activate = util.sigmoid
			self.loss_logit_form = util.logloss_logit_form

	def _init_history(self):
		self.history = {}
		self.history["current_iter"] = 0
		self.history["objective"] = np.zeros(self.config["max_iter"])
		self.history["train_logloss"] = np.zeros(self.config["max_iter"])
		self.history["test_logloss"] = np.zeros(self.config["max_iter"])
		self.history["train_logloss_no_noise"] = np.zeros(self.config["max_iter"])
		self.history["test_logloss_no_noise"] = np.zeros(self.config["max_iter"])
		self.history["train_time"] = np.zeros(self.config["max_iter"])
		# self.history["r_norm"] = np.zeros(self.config["max_iter"])
		# self.history["s_norm"] = np.zeros(self.config["max_iter"])

	def train(self):
		pass

	def train_privacy_test(self):
		pass

	def train_no_privacy(self):
		s_Q = np.zeros(self.server.train_size)
		z = np.zeros(self.server.train_size)
		y = np.zeros(self.server.train_size)
		for t in range(self.config["max_iter"]):
			time_start = time.time()
			# worker iteration
			for i in range(self.config["num_workers"]):
				self.worker[i].update_no_noise(s_Q, z, y)
			Q = []
			for i in range(self.config["num_workers"]):
				Q.append(self.worker[i].get_Dx_no_noise(is_train = True))
			s_Q = np.sum(Q, axis=0)
		# server iteration
			self.server.update(s_Q)
			z, y = self.server.get_z_y()
			time_end = time.time()
		# evaluation and print
			self.history["current_iter"] = t
			self.history["train_time"][t] = time_end - time_start
			self.history["train_logloss_no_noise"][t] = self.loss_logit_form(self.server.train_label, s_Q)
			self.history["test_logloss_no_noise"][t], _ = self.eval_test_no_privacy()
			if 0 != self.config["is_verbose"]:
				print("--Iteration: %d, train_logss: %6.4f, test_logloss: %6.4f, elapsed time: %4.2f" % (t, self.history["train_logloss_no_noise"][t], self.history["test_logloss_no_noise"][t], self.history["train_time"][t]))

	def eval_test_no_privacy(self):
		Dx = []
		for i in range(self.config["num_workers"]):
			Dx.append(self.worker[i].get_Dx_no_noise(is_train = False))
		sum_Dx = np.sum(Dx, axis=0)
		predict_proba = self.activate(sum_Dx)
		loss = self.loss_logit_form(self.server.test_label, sum_Dx)
		return loss, predict_proba

	def eval_test_privacy(self):
		pass

	def is_meet_stop(self):
		pass

if __name__ == "__main__":
	test = coord(config)
	test.train_no_privacy()


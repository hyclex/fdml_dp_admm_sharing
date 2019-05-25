## TO DO
# data loading and initialization
# training
# recording and saving
# noise adding and handling

from config import config
import os, sys
import numpy as np
import math
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy import sparse
import time
import pickle
if config["is_parallel"] == True:
	from multiprocessing import current_process, Pool 
	pool = Pool()

import util
# reload(util)
np.random.seed(0)

def fast_exp(x):
	pass

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
		self.num_iter = 0

	def update(self, s_Q):
		self.s_Q = s_Q
		self.z_old = self.z[:]
		self.update_z_onebyone()
		self.update_y()
		self.num_iter += 1

	def update_z(self):
		res = minimize(self.h, self.z, method=self.config["z_solver"], jac=self.h_der, hess=self.h_hess, options={'xtol': 1e-4, 'disp': False})
		self.z = res.x
	
	def update_z_onebyone(self):
		if True == self.config["is_parallel"]:
			self.cur_idx = {}
			pool.map(self._update_z_onebyone_para, range(self.train_size))
		else:
			for i in range(self.train_size):
				self.cur_idx = i
				res = minimize_scalar(self.h_per, options={'xtol': 1.e-2, 'maxiter': 10})
				# res = minimize_scalar(self.h_per)
				# res = minimize(self.h_per, self.z[i], method=self.config["z_solver"], jac=self.h_der_per, options={'disp': False})
				self.z[i] = res.x
				# res = root_scalar(self.h_der_per, x0=self.z[i], fprime=self.h_hess_per, method='newton')
				# self.z[i] = res.root

	def _update_z_onebyone_para(self, i):
		current = current_process()
		self.cur_idx[current._identity] = i
		res = minimize_scalar(self._h_per_para)
		self.z[i] = res.x


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
		zy = z*self.train_label
		return np.sum(np.log(1+np.exp(-zy)))

	def l_der(self, z):
		zy = z*self.train_label
		return -self.train_label / (1+np.exp(zy))

	def l_hess(self, z):
		zy = z*self.train_label
		ezy = np.exp(zy)
		return np.diag(self.train_label**2 * ezy / (1+ezy)**2)

	# to speedup try to optimize one by one since zs are not correlated
	def h_per(self, z):
		i = self.cur_idx
		l = self.l_per(z, i)
		linear = self.y[i] * z
		aug = self.config["rho"] * 2 * (self.s_Q[i] - z)**2
		return l - linear + aug

	def h_der_per(self, z):
		i = self.cur_idx
		l_der = self.l_der_per(z, i)
		return l_der - self.y[i] + self.config["rho"] * (z - self.s_Q[i])

	def h_hess_per(self, z):
		i = self.cur_idx
		l_hess = self.l_hess_per(z, i)
		return l_hess + self.config["rho"]


	def _h_per_para(self, z):
		current = current_process()
		i = self.cur_idx[current._identity]

		l = self.l_per(z, i)
		linear = self.y[i] * z
		aug = self.config["rho"] * 2 * (self.s_Q[i] - z)**2
		return l - linear + aug

	def l_per(self, z, i):
		zy = z*self.train_label[i]
		return math.log(1+math.exp(-zy))

	def l_der_per(self, z, i):
		zy = z*self.train_label[i]
		return -self.train_label[i] / (1+math.exp(zy))

	def l_hess_per(self, z, i):
		zy = z*self.train_label[i]
		ezy = math.exp(zy)
		# return self.train_label[i]**2 * ezy / (1+ezy)**2
		return  ezy / (1+ezy)**2 # train_label is always +-1

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

		# precompute the variables for privacy evaluation
		if "computed" == self.config["noise_eval_method"]:
			# to save time, try load the cache
			if True == self.config["is_cache_inverse_and_speed"]:
				cache_path = os.path.join(self.config["output_dir_path"], os.path.basename(self.config["input_dir_path"])+"_iDmTDmCache"+str(self.worker_id))
				try:
					with open(cache_path, "rb") as fin:
						self.i_DmTDm = pickle.load(fin)
				except:
					self.i_DmTDm = np.linalg.pinv((self.train_feature.transpose().dot(self.train_feature)).toarray())
					with open(cache_path, "wb") as fout:
						pickle.dump(self.i_DmTDm, fout)
			# print self.i_DmTDm.dot((self.train_feature.transpose().dot(self.train_feature)).toarray())
			self.sigma_const = np.sqrt(2*np.log(1.25/self.config["delta"])) / self.config["epsilon"]

	def update_no_noise(self, s_Q, z, y):
		self.s_Q = s_Q
		self.z = z
		self.y = y
		res = minimize(self.g, self.x, method=self.config["x_solver"], jac=self.g_der, hess=self.g_hess, options={'xtol': 1e-4, 'disp': False})
		self.x = res.x
		# self.Qx_old = self.Qx
		self.Qx = self.train_feature.dot(self.x.reshape([-1,1])).flatten()

	def update_and_append_noise(self, s_Q, z, y, method="result"):
		assert method in ["result", "variable"]
		self.update_no_noise(s_Q, z, y)
		# append noise
		noise_scale = self.eval_noise_scale()
		if "result" == method:
			assert "fixed" == self.config["noise_eval_method"]
			if "fixed" == self.config["noise_eval_method"]:
				self.Qx += np.random.normal(scale = noise_scale,size=[self.train_size])
		if "variable" == method:
			if "fixed" == self.config["noise_eval_method"]:
				self.Qx += self.train_feature.dot(np.random.normal(scale = noise_scale,size=[self.size_feature]))
			if "computed" == self.config["noise_eval_method"]:
				self.Qx += self.train_feature.dot(np.random.multivariate_normal(np.zeros(self.size_feature), noise_scale))

	def eval_noise_scale(self):	
		if "fixed" == self.config["noise_eval_method"]:
			return self.config["noise_scale"]
		if "computed" == self.config["noise_eval_method"]:
			C = 3. / self.size_feature / self.config["rho"] * (self.config["lambda"]*2 + np.linalg.norm(self.y) + self.config["rho"] * np.linalg.norm(self.s_Q - self.Qx))
			sigma = self.sigma_const * C
			# !!!
			print sigma**2
			return sigma**2 * self.i_DmTDm

	def get_x_norm(self):
		return np.linalg.norm(self.x)**2

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
		# R_hess = self.R_hess(x) # hess is I
		remain = self.config["rho"] * self.train_feature.transpose().dot(self.train_feature).toarray()
		return self.config["lambda"] + remain 

	def g_hess_sparse(self, x):
		R_hess = self.R_hess_sparse(x)
		# remain = self.config["rho"] * self.train_feature.transpose().dot(self.train_feature)
		return self.config["lambda"] * R_hess #+ remain

	def R(self, x):
		return np.linalg.norm(x)

	def R_der(self, x):
		return 2*x

	def R_hess(self, x):
		return 2*np.eye(len(x))

	def R_hess_sparse(self, x):
		dim = len(x)
		row = range(dim)
		col = range(dim)
		data = np.ones(dim)
		return 2 * sparse.csc_matrix((data, (row, col)), shape=(dim, dim))

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
		self.history["train_objective_no_noise"] = np.zeros(self.config["max_iter"])
		self.history["test_logloss_no_noise"] = np.zeros(self.config["max_iter"])
		self.history["train_time"] = np.zeros(self.config["max_iter"])
		self.history["train_time_x"] = np.zeros([self.config["max_iter"], self.config["num_workers"]])
		self.history["train_time_z"] = np.zeros(self.config["max_iter"])
		# self.history["r_norm"] = np.zeros(self.config["max_iter"])
		# self.history["s_norm"] = np.zeros(self.config["max_iter"])

	def train(self):
		s_Q = np.zeros(self.server.train_size)
		z = np.zeros(self.server.train_size)
		y = np.zeros(self.server.train_size)
		time_start = time.time()
		for t in range(self.config["max_iter"]):
			# worker iteration
			# if True == self.config["is_parallel"]:
				# Parallel(n_jobs=self.config["num_cpus"])(delayed(self.worker[i].update_no_noise)(s_Q, z, y) for i in range(self.config["num_workers"]))
			# else:
			if True == self.config["is_with_noise"]:
				for i in range(self.config["num_workers"]):
					time_xi_start = time.time()
					self.worker[i].update_and_append_noise(s_Q, z, y, method=self.config["noise_method"])
					self.history["train_time_x"][t, i] = time.time() - time_xi_start
			else:
				for i in range(self.config["num_workers"]):
					time_xi_start = time.time()
					self.worker[i].update_no_noise(s_Q, z, y)
					self.history["train_time_x"][t, i] = time.time() - time_xi_start
			time_z_start = time.time()
			Q = []
			for i in range(self.config["num_workers"]):
				Q.append(self.worker[i].get_Dx_no_noise(is_train = True))
			s_Q = np.sum(Q, axis=0)
		# server iteration
			self.server.update(s_Q)
			z, y = self.server.get_z_y()
			time_epoch_end = time.time()
		# evaluation and print
			self.history["current_iter"] = t
			self.history["train_time_z"][t] = time_epoch_end - time_z_start
			self.history["train_time"][t] = self.history["train_time_z"][t] + np.max(self.history["train_time_x"][t, :]) # simulate the true case, worker calculation is parallal 
			self.history["train_logloss_no_noise"][t] = self.loss_logit_form(self.server.train_label, s_Q)
			x_l2 = 0
			for i in range(self.config["num_workers"]):
				x_l2 += self.worker[i].get_x_norm()
			self.history["train_objective_no_noise"][t] = self.history["train_logloss_no_noise"][t] + self.config["lambda"] * x_l2
			self.history["test_logloss_no_noise"][t], _ = self.eval_test_no_privacy()
			if 0 != self.config["is_verbose"]:
				print("--Iteration: %d, train_logloss: %6.4f, train_obj: %6.4f, test_logloss: %6.4f, time(x, z): %4.2f/%4.2f, eplapsed: %4.2f, ETA: %4.2f" % (t, self.history["train_logloss_no_noise"][t], self.history["train_objective_no_noise"][t], self.history["test_logloss_no_noise"][t], np.max(self.history["train_time_x"][t, :]), self.history["train_time_z"][t], time_epoch_end-time_start, (time_epoch_end-time_start)/(t+1.)*(self.config["max_iter"]-t-1.)))
		if 0 != self.config["is_verbose"]:
			print("Total elapsed time %4.2f" % np.sum(self.history["train_time"]))

	def eval_test_no_privacy(self):
		Dx = []
		for i in range(self.config["num_workers"]):
			Dx.append(self.worker[i].get_Dx_no_noise(is_train = False))
		sum_Dx = np.sum(Dx, axis=0)
		predict_proba = self.activate(sum_Dx)
		loss = self.loss_logit_form(self.server.test_label, sum_Dx)
		return loss, predict_proba

	def save_history(self):
		if "fixed" == self.config["noise_eval_method"]:
			path = os.path.join(self.config["output_dir_path"], os.path.basename(self.config["input_dir_path"])+"_noise_"+str(self.config["noise_scale"]))
		if "computed" == self.config["noise_eval_method"]:
			path = os.path.join(self.config["output_dir_path"], os.path.basename(self.config["input_dir_path"])+"_epsilon_"+str(self.config["epsilon"])+"_delta_"+str(self.config["delta"]))
		with open(path, "wb") as fout:
			pickle.dump(self.history, fout)

	def eval_test_privacy(self):
		pass

	def is_meet_stop(self):
		pass

if __name__ == "__main__":
	# input_dir_path noise_scale num_workers
	if len(sys.argv) > 1:
		if sys.argv[1] == "h":
			print("input_dir_path noise_scale num_workers maxiteration [epsilon delta]")
			exit(1)
		print("Using command line parameters")
		try:
			config["input_dir_path"] = sys.argv[1]
			config["noise_scale"] = float(sys.argv[2])
			config["num_workers"] = int(sys.argv[3])
			config["max_iter"] = int(sys.argv[4])
		except:
			print("Wrong input parameters. Use option h for help.")
			exit(-1)
		if len(sys.argv) > 5:
			try:
				config["epsilon"] = float(sys.argv[5])
				config["delta"] = float(sys.argv[6])
				config["noise_eval_method"] = "computed"
			except:
				print("Wrong input parameters. Use option h for help.")
				exit(-1)
	else:
		print("Using default parameters in config")
	test = coord(config)
	test.train()
	test.save_history()

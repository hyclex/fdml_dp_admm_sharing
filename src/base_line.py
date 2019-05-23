from sklearn.linear_model import LogisticRegression
import time
import os, sys

import util
from config import config
# reload(util)


def test_lr(train_feature, train_label, test_feature, test_label):
	## build and train the model 
	lr_model = LogisticRegression(C = 1./config["lambda"])
	time_start = time.time()
	lr_model.fit(train_feature, train_label)
	time_end = time.time()
	time_elapsed = time_end - time_start
	# evaluate the result
	pred_proba_train = lr_model.predict_proba(train_feature)
	logloss_train = util.logloss(train_label, pred_proba_train)
	pred_proba_test = lr_model.predict_proba(test_feature)
	logloss_test = util.logloss(test_label, pred_proba_test)
	# print out
	print("Training logloss: %6.4f" % logloss_train)
	print("Training time: %6.2f" % time_elapsed)
	print("Testing logloss: %6.4f" % logloss_test)
	return time_elapsed, logloss_train, logloss_test

if __name__ == "__main__":
	data_set = "a9a"
	if len(sys.argv)>1:
		data_set = sys.argv[1]

	print("Evaluating "+data_set)
	result_dir = "../result/"
	result_path = os.path.join(result_dir, data_set+"_base_line")

	train_feature, train_label, test_feature, test_label = util.load_raw(data_set)
	full_time, full_train, full_test = test_lr(train_feature, train_label, test_feature, test_label)

	train_feature, train_label, test_feature, test_label = util.load_parts(data_set)
	loc_time, loc_train, loc_test = test_lr(train_feature[0], train_label[0], test_feature[0], test_label[0])

	with open(result_path, "w") as fout:
		fout.write("full time, full train loss, full test loss\n")
		fout.write(str(full_time)+","+str(full_train)+","+str(full_test)+"\n")
		fout.write("loc time, loc train loss, loc test loss\n")
		fout.write(str(loc_time)+","+str(loc_train)+","+str(loc_test)+"\n")


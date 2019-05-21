from sklearn.linear_model import LogisticRegression
import time

import util
from config import config
# reload(util)

def test_lr(train_feature, train_label, test_feature, test_label):
	## build and train the model 
	lr_model = LogisticRegression(C = 1./config["lambda"], solver="lbfgs")
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

if __name__ == "__main__":
	print("-----Evaluating a9a all...")
	train_feature, train_label, test_feature, test_label = util.load_a9a_raw()
	test_lr(train_feature, train_label, test_feature, test_label)

	print("-----Evaluating a9a part0...")
	train_feature, train_label, test_feature, test_label = util.load_a9a_parts()
	test_lr(train_feature[0], train_label[0], test_feature[0], test_label[0])

import scipy as sp 
import numpy as np 
import glob
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import os

def preprocess(input_path, milestones_in, is_shuffle=False):
    milestones = milestones_in[:]
    for file in glob.glob(input_path+"-*"):
        os.remove(file)
    [feature_raw, label] = load_svmlight_file(input_path)
    [num_row, num_col]= feature_raw.shape
    milestones.append(num_col)
    intervals = []
    for i in range(len(milestones)-1):
        intervals.append(np.arange(milestones[i], milestones[i+1]))
    if True==is_shuffle:
        feature_raw = feature_raw.toarray()
        np.random.seed(0)
        col_index = np.random.permutation(num_col)
        feature_raw = feature_raw[:, col_index]
        with open(input_path+"-random_index", "w") as f:
            for each in col_index:
                f.write(str(each)+" ")
    feature_coo = sp.sparse.coo_matrix(feature_raw)
    # generating and saving the parts
    query_id = np.arange(num_row)
    num_parts = len(intervals)
    for i in range(num_parts):
        # split the matrix
        index_merge = (feature_coo.col == intervals[i][0])
        for j in intervals[i]:
            index_merge = index_merge + (feature_coo.col == j)
        cur_feature_row = feature_coo.row[index_merge]
        cur_feature_col = feature_coo.col[index_merge]
        print "debug info:"
        print "col min", np.min(cur_feature_col)
        print "intervals min", np.min(intervals[i])
        cur_feature_col = cur_feature_col - np.min(intervals[i])
        cur_feature_data = feature_coo.data[index_merge]
        cur_feature = sp.sparse.coo_matrix((cur_feature_data, (cur_feature_row, cur_feature_col)), shape = (num_row, intervals[i].size))
        cur_output_path = input_path + "-part" + str(i)
        dump_svmlight_file(cur_feature, label, cur_output_path, query_id = query_id)
        cur_output_meta_path = cur_output_path + ".meta"
        with open(cur_output_meta_path, "w") as fout:
            fout.write(str(np.max(intervals[i]) - np.min(intervals[i]) + 1))        
    # generating the server parts
    # command = ["cut", "-d", "\" \"", "-f1,2", input_path+"-part0", ">", input_path+"-server"]
    command = ["cut", "-d", "\" \"", "-f1,2,3", input_path+"-part0", ">", input_path+"-server"] # for compatibility issue, added one more column
    command = " ".join(command)
    print command
    os.system(command)
    with open(input_path + "-server.meta", "w") as fout:
        fout.write("1")

if __name__ == "__main__":
    # change this for specific purpose
    input_dir_path = "../data/a9a"
    milestones = [0, 66]
    # input_dir_path = "../data/gisette"
    # milestones = [0, 3000]
    # input_dir_path = "../data/gisette"
    # milestones = [0, 2000, 4000]
    
    is_shuffle = False
    # for training data
    input_path = os.path.join(input_dir_path, "raw_train")
    preprocess(input_path, milestones, is_shuffle)
    # for testing data
    input_path = os.path.join(input_dir_path, "raw_test")
    preprocess(input_path, milestones, is_shuffle)


    

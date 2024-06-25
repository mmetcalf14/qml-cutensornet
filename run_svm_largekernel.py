import sys
import pathlib

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score



num_features = int(sys.argv[1])
reps = int(sys.argv[2])
gamma = float(sys.argv[3])
n_illicit_train = int(sys.argv[4])
n_licit_train = int(sys.argv[5])
n_illicit_test = int(sys.argv[6])
n_licit_test = int(sys.argv[7])
# This needs to be equal to the number of processors used to compute kernel mat
n_procs = int(sys.argv[8])

print("\nUsing the following parameters:")
print("")
print(f"\tn_procs: {n_procs}")
print("")
print(f"\tnum_features: {num_features}")
print(f"\treps: {reps}")
print(f"\tgamma: {gamma}")
print(f"\tn_illicit_train: {n_illicit_train}")
print(f"\tn_licit_train: {n_licit_train}")
print(f"\tn_illicit_test: {n_illicit_test}")
print(f"\tn_licit_test: {n_licit_test}")
print("")
sys.stdout.flush()

pathlib.Path("kernels").mkdir(exist_ok=True)


train_info = "train_Nf-{}_r-{}_g-{}_Ntr-{}".format(num_features, reps, gamma, n_illicit_train)
test_info = "test_Nf-{}_r-{}_g-{}_Ntr-{}".format(num_features, reps, gamma, n_illicit_test)


X_dim = n_licit_train + n_illicit_train
Y_dim = n_licit_test + n_illicit_test

entries_per_chunk = int(np.ceil(X_dim/n_procs))
print(n_procs*entries_per_chunk)
print('remainder ',n_procs*entries_per_chunk % X_dim)

kernel_train = np.zeros((X_dim,X_dim))
kernel_test = np.zeros((Y_dim,X_dim))
for rank in range(n_procs):
    print(rank)
    kernel_file_train = pathlib.Path(f"kernels/kernel_rank_{rank}_" + train_info + ".npy")
    kernel_file_test = pathlib.Path(f"kernels/kernel_rank_{rank}_" + test_info + ".npy")
    if kernel_file_train.is_file():
        tmp_train = np.load(kernel_file_train)
    else:
        raise ValueError(f"train kernel file for rank {rank} does not exist")
        
    if kernel_file_test.is_file():
        tmp_test = np.load(kernel_file_test)
    else:
        raise ValueError(f"test kernel file for rank {rank} does not exist")
    x_index_i = entries_per_chunk*rank
    if rank is not n_procs-1:
        x_index_j = entries_per_chunk*(rank+1)
    else: x_index_j = X_dim
    print(x_index_i,x_index_j)
    kernel_train[:, x_index_i:x_index_j] = tmp_train
    kernel_test[:,x_index_i:x_index_j] = tmp_test
    
kernel_train = np.transpose(kernel_train[:,0:X_dim]) + kernel_train[:,0:X_dim] - np.eye(X_dim,X_dim)
#print(kernel_train)
print(kernel_test)

############################
 #Testing the kernel matrix #
############################

###TODO Need to add the train labels back into the script to run SVM which requires the data sampling method...

#if rank == root:
#    reg = [2,1.5,1,0.5,0.1,0.05,0.01]
#    test_results = []
#    for key, r in enumerate(reg):
#        print('coeff: ', r)
#        svc = SVC(kernel="precomputed", C=r, tol=1e-5, verbose=False)
#        # scale might work best as 1/Nfeautres
#
#        svc.fit(kernel_train, train_labels)
#        test_predict = svc.predict(kernel_test)
#        accuracy = accuracy_score(test_labels,test_predict)
#        print('accuracy: ', accuracy)
#        precision = precision_score(test_labels,test_predict)
#        print('precision: ', precision)
#        recall = recall_score(test_labels, test_predict)
#        print('recall: ', recall)
#        auc = roc_auc_score(test_labels, test_predict)
#        print('auc: ', auc)
#        test_results.append([r,accuracy, precision, recall, auc])
#
#    train_results = []
#    print('\n Train Results\n')
#    for key, r in enumerate(reg):
#        print('coeff: ', r)
#        svc = SVC(kernel="precomputed", C=r, tol=1e-5, verbose=False)
#        # scale might work best as 1/Nfeautres
#
#        svc.fit(kernel_train, train_labels)
#        test_predict = svc.predict(kernel_train)
#        accuracy = accuracy_score(train_labels,test_predict)
#        print('accuracy: ', accuracy)
#        precision = precision_score(train_labels,test_predict)
#        print('precision: ', precision)
#        recall = recall_score(train_labels, test_predict)
#        print('recall: ', recall)
#        auc = roc_auc_score(train_labels, test_predict)
#        print('auc: ', auc)
#        train_results.append([r,accuracy, precision, recall, auc])
#
#    np.save('data/TrainData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit_train),train_results)
#    np.save('data/TestData_Nf-{}_r-{}_g-{}_Ntr-{}.npy'.format(num_features, reps, gamma, n_illicit_test),test_results)

import numpy as np
import matplotlib.pyplot as plt

# Gather results from data size 300
traindata_15q = np.load("raw/train_Nf15_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
traindata_50q = np.load("raw/train_Nf50_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
traindata_100q = np.load("raw/train_Nf100_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
traindata_165q = np.load("raw/train_Nf165_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
testdata_15q = np.load("raw/test_Nf15_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
testdata_50q = np.load("raw/test_Nf50_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
testdata_100q = np.load("raw/test_Nf100_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")
testdata_165q = np.load("raw/test_Nf165_r2_g0.1_p0.0_nn1_mslinear_Ntr150_s5_elliptic_preproc.npy")

best_train_auc_small = [np.max(traindata_15q[:,4]),np.max(traindata_50q[:,4]),np.max(traindata_100q[:,4]),np.max(traindata_165q[:,4])]
best_test_auc_small = [np.max(testdata_15q[:,4]),np.max(testdata_50q[:,4]),np.max(testdata_100q[:,4]),np.max(testdata_165q[:,4])]

# Gather results from data size 1500
traindata_15q = np.load("raw/train_Nf15_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
traindata_50q = np.load("raw/train_Nf50_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
traindata_100q = np.load("raw/train_Nf100_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
traindata_165q = np.load("raw/train_Nf165_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
testdata_15q = np.load("raw/test_Nf15_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
testdata_50q = np.load("raw/test_Nf50_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
testdata_100q = np.load("raw/test_Nf100_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")
testdata_165q = np.load("raw/test_Nf165_r2_g0.1_p0.0_nn1_mslinear_Ntr750_s5_elliptic_preproc.npy")

best_train_auc_med = [np.max(traindata_15q[:,4]),np.max(traindata_50q[:,4]),np.max(traindata_100q[:,4]),np.max(traindata_165q[:,4])]
best_test_auc_med = [np.max(testdata_15q[:,4]),np.max(testdata_50q[:,4]),np.max(testdata_100q[:,4]),np.max(testdata_165q[:,4])]

# Gather results from data size 6400
traindata_15q = np.load("raw/train_Nf15_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
traindata_50q = np.load("raw/train_Nf50_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
traindata_100q = np.load("raw/train_Nf100_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
traindata_165q = np.load("raw/train_Nf165_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
testdata_15q = np.load("raw/test_Nf15_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
testdata_50q = np.load("raw/test_Nf50_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
testdata_100q = np.load("raw/test_Nf100_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")
testdata_165q = np.load("raw/test_Nf165_r2_g0.1_p0.0_nn1_mslinear_Ntr3200_s5_elliptic_preproc.npy")

best_train_auc_large = [np.max(traindata_15q[:,4]),np.max(traindata_50q[:,4]),np.max(traindata_100q[:,4]),np.max(traindata_165q[:,4])]
best_test_auc_large = [np.max(testdata_15q[:,4]),np.max(testdata_50q[:,4]),np.max(testdata_100q[:,4]),np.max(testdata_165q[:,4])]

fig, ax = plt.subplots()
q = [15,50,100,165] # Number of qubits
ax.plot(q, best_train_auc_large, marker='o', color='blue', label='6400')
ax.plot(q, best_train_auc_med,marker='o', color = 'goldenrod', label='1500')
ax.plot(q, best_train_auc_small,marker='o', color = 'green', label='300')
ax.set(xlim=(0,165),ylim=(0.5,1))
ax.legend(loc='lower right', title="Data Size")
ax.set_xlabel('Number of Features')
ax.set_ylabel('AUC')

plt.show()

fig, ax = plt.subplots()
q = [15,50,100,165] # Number of qubits
ax.plot(q, best_test_auc_large, marker='o', color='blue', label='6400')
ax.plot(q, best_test_auc_med,marker='o', color = 'goldenrod', label='1500')
ax.plot(q, best_test_auc_small,marker='o', color = 'green', label='300')
ax.set(xlim=(0,165),ylim=(0.5,1))
ax.legend(loc='lower right', title="Data Size")
ax.set_xlabel('Number of Features')
ax.set_ylabel('AUC')

plt.show()

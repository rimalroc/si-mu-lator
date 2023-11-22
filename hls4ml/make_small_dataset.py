import numpy as np
import matplotlib as mpl
import numpy as np
import h5py
import sys
sys.path.append('../algorithms/')

import datatools
import trainingvariables
from glob import glob
from sklearn.utils import shuffle

files_loc = "/Data/ML/si-mu-lator/simulation_data"
fdir = "SIG_atlas_nsw_pad_z0_xya"
nevs=50000
do_det_matrix=True
#CARD='atlas_nsw_pad_z0_stg2BC'
#CARD='atlas_nsw_pad_z0_stg300um'
CARD='atlas_nsw_pad_z0_mm4BC'
det_card=f'../cards/{CARD}.yml'


all_files = glob(f'/Data/ML/si-mu-lator/simulation_data/{CARD}_bkgr_1'+'/VALIDATE/W*.h5')
print(all_files)
data, dmat, Y, Y_mu, Y_hit, sig_keys = datatools.make_data_matrix(all_files, max_files=10, sort_by='z')

this_cut=(Y_mu==1)

if do_det_matrix:
    X_prep = datatools.detector_matrix_2(dmat, sig_keys, det_card)
else:
    X_prep = datatools.training_prep(dmat, sig_keys)

vars_of_interest = np.zeros(X_prep.shape[2], dtype=bool)
training_vars = trainingvariables.tvars
for tv in training_vars:
    vars_of_interest[sig_keys.index(tv)] = 1
X = X_prep[:,:,vars_of_interest]

mult_fact_X = max(data['ev_mu_x'])
mult_fact_a = max(data['ev_mu_theta'])
print(f"#&#&#&#&#&#&# X mult fact = {mult_fact_X}, Angle mult fact = {mult_fact_a} #&#&#&#&#&#&#")
data_ev_mu_x = (data['ev_mu_x'])/mult_fact_X
data_ev_mu_a = (data['ev_mu_theta'])/mult_fact_a

X_test, Y_clas_test, Y_xreg_test, Y_areg_test = shuffle(X, Y_mu, data_ev_mu_x, data_ev_mu_a)

Y_test = np.zeros( (Y_clas_test.shape[0], 2 ) )
Y_test[:,0] = Y_xreg_test
Y_test[:,1] = Y_areg_test

out_name_tag = f"test_{nevs}_padMat_{fdir}.npy"
if do_det_matrix:
    out_name_tag = f"test_{nevs}_detMat_{CARD}.npy"


np.save(f"X_{out_name_tag}", X_test[this_cut][:nevs])
np.save(f"Y_{out_name_tag}", Y_test[this_cut][:nevs])

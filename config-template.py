import os
ROOT_DIR=os.path.realpath(__file__).replace("/config.py","")
#Data location
SIM="/Data/ML/si-mu-lator/simulation_data"
#detector card name in ROOT_DIR/cards
CARD="atlas_nsw_pad_z0"
#CARD="atlas_nsw_pad_z0_stg2BC"
#CARD="atlas_nsw_pad_z0_stg300um"
#CARD="atlas_nsw_pad_z0_mm4BC"
#use (1) or not use (0) data with background
BKGR="1"
DATA_LOC=f"{SIM}/{CARD}_bkgr_{BKGR}/*.h5"
DATA_LOC_TEST=f"{SIM}/{CARD}_bkgr_{BKGR}/TEST/W*.h5"
DATA_LOC_TRAIN=f"{SIM}/{CARD}_bkgr_{BKGR}/TRAIN/W*.h5"
DATA_LOC_VALIDATE=f"{SIM}/{CARD}_bkgr_{BKGR}/VALIDATE/*.h5"
SING_IMG="/Data/images/hls4ml_sandbox"
SING_IMG_TRAIN="/Data/images/si-mu-lator_train"
DET=f"{ROOT_DIR}/cards/{CARD}.yml"
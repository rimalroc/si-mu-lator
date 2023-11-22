#!/bin/bash

source ../config.sh 
#SIM="/Data/ML/si-mu-lator/simulation_data"
#CARD="atlas_nsw_pad_z0"
#DATA_LOC="${SIM}/${CARD}_bkgr_1/*.h5"
#SING_IMG=/Data/images/slac-ml@20211101.0.sif                                                 
#SING_IMG=/Data/images/hls4ml_sandbox
#DET="../cards/${CARD}.yml"


#singularity exec -B /Data ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification" --detector ${DET}

#SIM="/gpfs/slac/atlas/fs1/d/rafaeltl/public/Muon/simulation/20220912/SIG_atlas_nsw_pad_z0_xya/"
#DATA_LOC="${SIM}/TRAIN/*.h5"
#SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/muon_qkeras.sif

#VALS_CBNORM=( "--conv-batchnorm  --input-batchnorm" " --conv-batchnorm " " ")
#VALS_CBNORM=( "--conv-batchnorm --batchnorm-constraint" " " )
VALS_CBNORM=( " --conv-batchnorm" )
#VALS_DBNORM=(  " --dense-batchnorm " )
VALS_DBNORM=( " " )
VALS_BIAS=( "" )
#VALS_PEN=("--no-pen")
VALS_PEN=( "" )
VALS_PTYPE=( 0 )
VALS_BKGPEN=( " " )

#VALS_CLAYERS=( "7,1,1,0:5,3,1,0" "7,1,1,0:5,4,1,0" "7,1,1,0:5,5,1,0" "7,1,1,0:10,3,1,0" "7,1,1,0:10,4,1,0" "7,1,1,0:10,5,1,0" "7,1,1,0:20,3,1,0" "7,1,1,0:20,4,1,0" "7,1,1,0:20,5,1,0" "5,3,1,0" "5,4,1,0" "5,5,1,0" "10,3,1,0" "10,4,1,0" "10,5,1,0"  "15,3,1,0" "15,4,1,0" "15,5,1,0" "20,3,1,0" "20,4,1,0" "20,5,1,0" "5,3,1,0:5,3,1,0" "5,4,1,0:5,4,1,0" "5,5,1,0:5,5,1,0" "7,1,1,0:25,3,1,0" "7,1,1,0:25,4,1,0" "7,1,1,0:25,5,1,0" "7,1,1,0:30,3,1,0" "7,1,1,0:30,4,1,0" "7,1,1,0:30,5,1,0" "25,3,1,0" "25,4,1,0" "25,5,1,0" "30,3,1,0" "30,4,1,0" "30,5,1,0" "7,4,4,0:5,3,1,0" "7,4,4,0:5,4,1,0" "7,4,4,0:5,5,1,0" "7,4,4,0:10,3,1,0" "7,4,4,0:10,4,1,0" "7,4,4,0:10,5,1,0" "7,4,4,0:20,3,1,0" "7,4,4,0:20,4,1,0" "7,4,4,0:20,5,1,0" "7,4,4,0:25,3,1,0" "7,4,4,0:25,4,1,0" "7,4,4,0:25,5,1,0" "7,4,4,0:30,3,1,0" "7,4,4,0:30,4,1,0" "7,4,4,0:30,5,1,0" "7,1,1,0:7,6,6,0" "7,1,1,0:7,5,5,0" "7,1,1,0:7,4,4,0" "7,1,1,0:7,3,3,0" "7,1,1,0:7,2,2,0" "5,3,1,0:6,3,1,0" "7,3,1,0:5,3,1,0" "5,3,1,0:5,3,1,0" "5,4,1,0:5,4,1,0" "4,3,1,0:4,3,1,0" "5,3,1,0:5,3,3,0" "5,4,1,0:5,4,3,0" "4,3,1,0:4,3,3,0" )
#"7,1,1,0:5,3,1,0"
#VALS_CLAYERS=( "7,1,1,0:5,3,1,0" "25,3,1,0" )
#VALS_CLAYERS=( "7,1,1,0:5,3,1,0" )
#VALS_CLAYERS=( "4,3,1,0:4,3,3,0" "7,1,1,0:5,3,1,0")
#VALS_CLAYERS=( "5,4,4,0:4,3,1,0" "10,4,4,0:10,3,1,0")
VALS_CLAYERS=( "10,4,4,0:10,3,1,0" )
#VALS_DLAYERS=(  "none" "10" "20" "50" )
#VALS_DLAYERS=(  "20" "20:4")
VALS_DLAYERS=(  "20"  "20" "20" "20" "20" "20" "20" "20" "20" "20")
#VALS_DLAYERS=(  "none" )

POOL=( "--flatten"  )

#QK="--do-q"
QK=" "

# QKBS=( 4 6 8 10 12 )
# QIKBS=( 0 2 4 6 )
QKBS=( 22 )
QIKBS=( 5 )

#L1REG=" "
L1REG="--l1reg 0.0005"

#LRATE=( "0.0005" "0.001" "0.005" )

ijob=400

#DCARD=" "
DCARD=" --detmat ${DET}"
PRUNES=( " " "--prune 50")
#PRUNES=( " " )
#BIG_MOD="--bigger-model /afs/cern.ch/user/r/rrojas/public/ML/r-dev-branch/si-mu-lator/algorithms/models/MyTCN_CL7.1.1.0..5.3.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run18"

#BIG_MOD="--bigger-model /afs/cern.ch/user/r/rrojas/public/ML/r-dev-branch/si-mu-lator/algorithms/models/MyTCN_CL7.1.1.0..5.3.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run24"

if [ $(hostname) == "umasscastor1n1" ]; then
    FLAG="--flag c1n1"
elif [ $(hostname) == "umassminipc02" ]; then
    FLAG="--flag pc02"
elif [ $(hostname) == "umasscastor1n2" ]; then
    FLAG="--flag c1n2"
else
    FLAG=""
fi;


for cbnorm in "${VALS_CBNORM[@]}"
do
    for dbnorm in "${VALS_DBNORM[@]}"
    do
        for rbias in "${VALS_BIAS[@]}"
        do
            for pen in "${VALS_PEN[@]}"
            do
                for ptype in "${VALS_PTYPE[@]}"
                do
                    for bkgpen in "${VALS_BKGPEN[@]}"
                    do
                        for clayers in "${VALS_CLAYERS[@]}"
                        do
                            for dlayers in "${VALS_DLAYERS[@]}"
                            do
                                for pool in "${POOL[@]}"
                                do
                                    for QKB in "${QKBS[@]}"
                                    do
                                        for QIKB in "${QIKBS[@]}"
                                        do
                                            for PRUNE in "${PRUNES[@]}"
                                            do
                                                echo ${cbnorm} ${dbnorm} ${lambda} ${pen} ${ptype} ${bkgpen} ${clayers} ${dlayers} ${rbias} ${pool} ${QKB} ${QIKB} ${PRUNE}
                                            # jname="TCN_${cbnorm}_${dbnorm}_${lambda}_${pen}_${ptype}_${bkgpen}_${clayers}_${dlayers}_${rbias}"
                                                jname="TCNjob${ijob}"
                                                ((ijob=ijob+1))
                                                COMM="singularity exec -B /Data -B /afs ${SING_IMG_TRAIN} python train.py -f \"${DATA_LOC_TRAIN}\" ${cbnorm} ${dbnorm} --lambda ${lambda} ${pen} --pen-type ${ptype} ${bkgpen} --clayers ${clayers} --dlayers ${dlayers} ${rbias} ${pool} ${QK} ${RES} --q-bits ${QKB} --q-ibits ${QIKB} ${L1REG} ${DCARD} ${FLAG} ${BIG_MOD} ${PRUNE}"
                                                echo "# INFO: CMD is > ${COMM}"
                                                /afs/cern.ch/user/r/rrojas/private/Inform_to_rrojas_message_telegram.sh "INFO: \`$(date)\`: Starting training of  $jname at \`$FLAG\` "
    #singularity exec -B /Data ${SING_IMG} python train.py -f "${DATA_LOC}" -m "tcn" --task "classification" --detector ${DET}
                                                singularity exec -B /Data -B /afs ${SING_IMG_TRAIN} python train.py -f "${DATA_LOC_TRAIN}" ${cbnorm} ${dbnorm} ${pen} --pen-type ${ptype} --clayers ${clayers} --dlayers ${dlayers} ${rbias} ${pool} ${QK} --q-bits ${QKB} --q-ibits ${QIKB} ${L1REG} ${DCARD} ${FLAG}"_"${ijob} ${BIG_MOD} ${PRUNE}

                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

/afs/cern.ch/user/r/rrojas/private/Inform_to_rrojas_message_telegram.sh "INFO: \`$(date)\`: Training at \`$FLAG\` finished "

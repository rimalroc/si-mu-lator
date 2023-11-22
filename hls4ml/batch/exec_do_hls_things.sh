#!/bin/bash
# IMG_NAME_TCN=muon_hls4ml_05072022.sif
#IMG_NAME_TCN=muon_hls4ml_qkeras.sif
#IMG_NAME_RNN=rnn_hls_keras_rnn_staticswitch_Apr25.sif
source ../../config.sh 
#SING_IMG=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/${IMG_NAME_TCN}
#SING_IMG=/Data/images/hls4ml_sandbox 
#SING_IMG=/Data/images/rnn_hls_keras_rnn_staticswitch_Apr25.sif
#fra_width=( 2 4 6 8 10 12 14 16 18 20 )
#int_width=( 0 1 2 3 4 6 )
#r_factor=( 1 2 5 )

stat=1 #default==1
#strat=( "Resource" "Latency"  )
strat=( "Latency" )
# strat=( "Resource" )

#rewrite=false
rewrite=false
#fra_width=( 2 )
int_width=( 2 4 6 )
#int_width=( 12 )
#r_factor=( 12 )

fra_width=(  12 14 16 )
#fra_width=( 16 )
#int_width=( 2 4 6 )
r_factor=( 1 2 5 )
#r_factor=( 5 )

#MOD_LOC=/gpfs/slac/atlas/fs1/u/rafaeltl/Muon/21062022/si-mu-lator/algorithms/models/
MOD_LOC=/afs/cern.ch/user/r/rrojas/public/ML/r-dev-branch/si-mu-lator/algorithms/models

# MOD="MyTCN_CL5.3.1.0..5.3.3.0_DLnone_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
# MOD="MyTCN_CL20.4.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
# MOD="MyTCN_CL7.4.4.0..5.3.1.0_DLnone_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
# MOD="MyTCN_CL4.3.1.0..4.3.3.0_DLnone_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
#MOD="MyTCN_CL7.3.1.0..5.3.1.0_DLnone_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_DetMat_4Outputs"
#MOD="MyTCN_CL4.3.1.0..4.3.3.0_DLnone_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0010_DetMat_pc02_4Outputs_LONG"
MODS=( "MyTCN_CL4.3.1.0..4.3.3.0_DLnone_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0010_DetMat_pc02_4Outputs_LONG_run15" "MyTCN_CL4.3.1.0..4.3.3.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0010_DetMat_pc02_4Outputs_LONG_run15" "MyTCN_CL7.1.1.0..5.3.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0010_DetMat_pc02_4Outputs_LONG_run11" "MyTCN_CL4.3.1.0..4.3.3.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run20" )
#MODS=( "MyTCN_CL4.3.1.0..4.3.3.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run20" )
MODS=( "MyTCN_CL7.1.1.0..5.3.1.0_DL20_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run25" )
MODS=( "QKeras.b16_QKeras.i2_MyTCN_CL7.1.1.0..5.3.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run34" "QKeras.b16_QKeras.i2_MyTCN_CL7.1.1.0..5.3.1.0_DL20_CBNormFalse_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_run34" )
MODS=( "MyTCN_CL10.4.4.0..10.3.1.0_DL20_CBNormTrue_DBNormFalse_IBNormFalse_penTrue_ptype0_regBiasTrue_lrate0.001_Flatten_L1R0.0005_DetMat_pc02_4Outputs_LONG_PRUNED50_run38" )
echo "HERE"

DATALOC=/afs/cern.ch/user/r/rrojas/public/ML/r-dev-branch/si-mu-lator/hls4ml/
DATA="${DATALOC}/X_test_50000_detMat_atlas_nsw_pad_z0.npy,${DATALOC}/Y_test_50000_detMat_atlas_nsw_pad_z0.npy"

OUTD=/Data/ML/models/last

/afs/cern.ch/user/r/rrojas/private/Inform_to_rrojas_message_telegram.sh "INFO: \`$(date)\`: HLS4ML task has begun"
jname=name

for fw in "${fra_width[@]}"
do
    for iw in "${int_width[@]}"
    do
        for rf in "${r_factor[@]}"
        do
            for st in "${strat[@]}"
            do
                for MOD in "${MODS[@]}"
                do

                jname=${MOD}_${fw}_${iw}_${rf}_${st}
            
                # singularity exec -B /sdf,/gpfs,/scratch ${SING_IMG} python do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} --fwidth ${fw} --iwidth ${iw}  -r ${rf} --vivado --static ${stat} --strategy ${strat}  --lut

                # break 91382098

#                sbatch --partition=usatlas \
#                    --job-name=${jname} --output=out/${jname}_o.txt \
#                    --error=err/${jname}_e.txt --ntasks=1 \
#                    --cpus-per-task=4 --mem-per-cpu=10g \
#                    --time=1:00:00 \
#                    << EOF
##!/bin/sh
if [ $rewrite = false ];
then
    if [ -f $OUTD/$MOD/reports/model_$(( $fw + $iw )).${iw}_reuse_${rf}_Latency_Static_BigTable.txt ]; then
        echo "INFO: skipping model $jname because rewrite is disabled"
        continue
    fi;
fi;
singularity exec -B /Data,/tools:/opt ${SING_IMG} python do_hls_things.py --name ${MOD} -a ${MOD_LOC}/${MOD}/arch.json -w ${MOD_LOC}/${MOD}/weights.h5 -d "${DATA}" -o ${OUTD} --fwidth ${fw} --iwidth ${iw}  -r ${rf} --vivado --static ${stat} --strategy ${st}  --lut
#EOF

                done
            done
        done
    done
    /afs/cern.ch/user/r/rrojas/private/Inform_to_rrojas_message_telegram.sh "INFO: \`$(date)\`: MOD \`$jname\` done"
done

/afs/cern.ch/user/r/rrojas/private/Notify_to_rrojas_message_telegram.sh "INFO: \`$(date)\`: HLS4ML task is done"
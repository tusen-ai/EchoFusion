DIR=kradar
WORK=work_dirs
CONFIG="ra_img_echofusion_kradar_r50_trainval_24e"
bash tools/dist_train.sh projects/configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/
sleep 60

# Evaluation
NUM=24
bash tools/dist_test.sh projects/configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_$NUM.pth 8 --eval bbox
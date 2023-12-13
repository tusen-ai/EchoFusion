DIR=radial
WORK=work_dirs
CONFIG="rt_img_echofusion_r50_da_24e"
bash tools/dist_train.sh projects/configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/

sleep 60

# Evaluation
NUM=24
bash tools/dist_test.sh projects/configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_$NUM.pth 8 --eval bbox
python tools/eval_radial.py --save-folder ./$WORK/$CONFIG --save-suffix $NUM

NUM=23
bash tools/dist_test.sh projects/configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_$NUM.pth 8 --eval bbox
python tools/eval_radial.py --save-folder ./$WORK/$CONFIG --save-suffix $NUM

NUM=22
bash tools/dist_test.sh projects/configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/epoch_$NUM.pth 8 --eval bbox
python tools/eval_radial.py --save-folder ./$WORK/$CONFIG --save-suffix $NUM
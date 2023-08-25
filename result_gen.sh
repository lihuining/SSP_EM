
# MOT17
python3 result_gen.py --path /media/allenyljiang/2CD8318DD83155F4/04_Dataset/Multi_Object_Tracking/MOT17 -f exps/example/mot/yolox_x_mot17_half.py --eval test -c pretrained/bytetrack_x_mot17.pth.tar --benchmark MOT17
# MOT16
python3 result_gen.py --path /media/allenyljiang/2CD8318DD83155F4/04_Dataset/Multi_Object_Tracking/MOT16 -f exps/example/mot/yolox_x_mot17_half.py --eval test -c pretrained/bytetrack_x_mot17.pth.tar --benchmark MOT16
# MOT20
python3 result_gen.py --path /media/allenyljiang/564AFA804AFA5BE5/Dataset/MOT20 -f exps/example/mot/yolox_x_mix_mot20_ch.py --eval test -c pretrained/bytetrack_x_mot20.tar --benchmark MOT20

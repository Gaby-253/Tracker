1. Film the stuff you want

2. Upload on roboflow for annotations https://app.roboflow.com/

3. Annotate (roboflow)

4. Create Dataset (roboflow)

5. Export in whatever format (yolov11) put in under dataset folder

6. Install yolo (preferably on a virtual python env) like conda :

conda create -n ant_yolo python=3.10 -y

conda activate ant_yolo

pip install ultralytics opencv-python pandas matplotlib numpy

yolo checks

7. download the model .pt at https://docs.ultralytics.com/models/yolo11#overview

6. train using : python code/train_seg.py

7. test with : python code/track_seg_centroid.py

It will track a video and output a .csv of the ant position in cm (you have to specifiy the px/cm scale in the code)


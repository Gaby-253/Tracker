from ultralytics import YOLO

model = YOLO("model/yolo26s-seg.pt")  # use yolo11n-seg.pt if GPU memory is low

model.train(
    data="dataset/proprio_ata/annotated/data.yaml",
    epochs=100,
    imgsz=2016,
    batch=1,
    task="segment",
    project="runs/segment",
    name="ata_ant_seg_train",
    amp=True,
)
from ultralytics import YOLO

model = YOLO("model/yolo26m-seg.pt")  # use yolo11n-seg.pt if GPU memory is low

model.train(
    data="dataset/proprio_ata/annotated/data.yaml",
    epochs=100,
    imgsz=1024,
    batch=4,
    task="segment",
    project="runs/segment",
    name="ata_ant_seg_train",
    amp=True,
)
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------- CONFIG ----------
model_path = "best_weights/best.pt"
video_path = "dataset/proprio_ata/raw/C0201.MP4"

output_dir = "tracking_outputs"
output_video_name = "proprio_ata_track_test2.mp4"
output_csv_name = "proprio_ata_track_test.csv"

imgsz = 2016 #Check the model training image size
conf = 0.25
pixel_to_cm = 0.07  # example: 25 px = 0.5 cm
# ----------------------------

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_path = os.path.join(output_dir, output_video_name)
csv_path = os.path.join(output_dir, output_csv_name)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

rows = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    annotated = frame.copy()

    if result.masks is not None and len(result.masks.data) > 0:
        # choose first detected ant
        ant_idx = 0

        mask = result.masks.data[ant_idx].cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask_bin = (mask > 0.5).astype("uint8")

        M = cv2.moments(mask_bin)

        if M["m00"] > 0:
            cx_px = M["m10"] / M["m00"]
            cy_px = M["m01"] / M["m00"]

            cx_cm = cx_px * pixel_to_cm
            cy_cm = cy_px * pixel_to_cm

            rows.append({
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "x_centroid_px": cx_px,
                "y_centroid_px": cy_px,
                "x_centroid_cm": cx_cm,
                "y_centroid_cm": cy_cm,
            })

            # draw mask overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 1] = mask_bin * 255
            annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.35, 0)

            # draw centroid
            cv2.circle(
                annotated,
                (int(cx_px), int(cy_px)),
                6,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                annotated,
                f"centroid: ({cx_cm:.2f}, {cy_cm:.2f}) cm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    writer.write(annotated)
    frame_idx += 1

cap.release()
writer.release()

df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

print(f"Saved video: {out_path}")
print(f"Saved CSV: {csv_path}")
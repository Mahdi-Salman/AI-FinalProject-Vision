from ultralytics import YOLO
import torch
import os

def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f" Training on device: {device}")

    print(" Building Custom YOLOv8 Model from Scratch...")
    model = YOLO('configs/custom_yolov8.yaml') 


    print(" Starting Training for 8 Classes...")
    results = model.train(
        data='configs/kitti.yaml',
        epochs=100,
        patience=15,
        batch=16,
        imgsz=640,
        device=device,
        optimizer='SGD',
        lr0=0.001, 
        project='runs/train',
        name='kitti_custom_8class', 
        exist_ok=True, 
        verbose=True
    )

    print(" Training Finished!")

if __name__ == '__main__':
    main()
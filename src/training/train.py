from ultralytics import YOLO

def train_advanced_model():
    print(" Starting Advanced Training with Custom YOLOv8s...")

    model = YOLO('configs/custom_yolov8.yaml')
    model.load('yolov8s.pt')
    
    model.train(
        data='configs/kitti.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        patience=10,
        mosaic=1.0,
        mixup=0.1,
        project='runs/train',
        name='YOLOv8s_Advanced_Run'
    )
    
    print(" Advanced Training Completed Successfully!")

if __name__ == '__main__':
    train_advanced_model()
from ultralytics import YOLO
import os

def evaluate_baseline():
    print("Starting Evaluation on Unseen Test Set...")
    
    weights_path = 'runs/detect/runs/train/kitti_custom_8class/weights/best.pt'
    
    if not os.path.exists(weights_path):
        print(f"Error: Model not found at {weights_path}")
        return

    model = YOLO(weights_path)
    
    print("\n📊 Evaluating BASELINE model...")
    metrics = model.val(
        data='configs/kitti.yaml',
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        project='runs/evaluate',
        name='baseline_test' 
    )
    
    print("\n Baseline Evaluation Completed!")
    print(f" Baseline mAP50-95 : {metrics.box.map:.4f}")
    print(f" Baseline mAP50    : {metrics.box.map50:.4f}")

if __name__ == '__main__':
    evaluate_baseline()
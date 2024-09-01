from ultralytics import YOLO
import torch
import os

def _environ_bool(key: str) -> bool:
    val = os.environ[key]
    return val == "True" or val == "true"


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    resume_path = os.environ["RESUME_PATH"]
    resume = resume_path != "None"

    if resume:
        model = YOLO(resume_path)
    else:
        model = YOLO(os.environ["MODEL_VERSION"])

    model.train(data=os.environ['CONFIG_PATH'], epochs=int(os.environ['EPOCHS']), dropout=float(os.environ['DROPOUT']), imgsz=(int(os.environ['IMG_WIDTH']), 
                int(os.environ['IMG_HEIGHT'])), device=device, batch=int(os.environ['BATCH_SIZE']), project=os.environ['PROJECT_PATH'], resume=resume,
                patience=int(os.environ['PATIENCE']), overlap_mask=_environ_bool('OVERLAP_MASK'), mask_ratio=int(os.environ['MASK_RATIO']))

if __name__ == "__main__":
    main()
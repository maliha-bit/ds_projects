# Project 3: Plant Disease Classification (Transfer Learning)

Classifies plant leaf images into 38 disease/healthy categories using a pretrained ResNet18.

## Dataset
PlantVillage (from Kaggle): 38 classes of crop leaf images. The `train` folder was used, and the script creates its own 80/20 train/validation split.

Because of limited hardware, only **60 images per class** were used (about 2,300 images in total).

## Method
- Pretrained ResNet18 (ImageNet weights) with the backbone frozen and only the final layer trained.
- Data augmentation: random flip, rotation, color jitter.
- Trained for 2 epochs with the Adam optimizer.
- Evaluation: validation accuracy, per-class report, confusion matrix, and a list of misclassified images for error analysis.

## Results

| Setup | Validation accuracy |
|---|---|
| ResNet18, frozen backbone, 60 images/class, 2 epochs | 71% |

For comparison, random guessing over 38 classes would give about 3%. The score is limited by the small subset and very short training. Fine-tuning deeper layers, using more images per class, and training for more epochs would very likely improve it substantially.

Outputs (classification report, confusion matrix, misclassified images list) are in `outputs_p3/`.

## Limitations
- Small subset and only 2 epochs due to limited compute (CPU only).
- Backbone frozen; the `--finetune` option was not used.
- PlantVillage images are taken in controlled conditions, so real-world field performance would likely be lower.

## How to run
Download PlantVillage and place it in `data/PlantVillage/` (the folder containing one sub-folder per class is what you pass).
```
pip install -r requirements.txt
python p3_plant_disease_transfer.py --data_dir data/PlantVillage/train --epochs 2 --max_per_class 60
```
Add `--finetune` and increase `--max_per_class` for better accuracy.

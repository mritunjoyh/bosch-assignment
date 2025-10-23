# BOSCH Assignment — Setup Instructions

This repository provides the code, environment, and dataset setup required for the BOSCH Assignment.

Follow the instructions below to either **use the pre-built Docker image** or **build it manually**.

---

## Option 1 — Use Pre-Built Docker Image (Recommended)

the pre-built Docker image (`.tar.gz`, ~10 GB) can directly be downloaded from Google Drive and load it into Docker.

```bash
# Download pre-built Docker image (~10GB)
# It can also be directly downloaded from the link.
gdown https://drive.google.com/file/d/19M1Mah8p9qK3SAjuLPAW2YBHoViEoIoz/view?usp=drive_link --fuzzy

# Load the image into Docker
sudo docker load < bosch-assign-bdd.tar.gz

# Verify that the image was loaded
sudo docker images
```
Once loaded, the container directly can be run :
```bash
sudo docker run -it --name temp bosch-assign-bdd bash
```
Note:
The image is large (~10 GB). Ensure enough disk space and a stable network connection.

---
## Option 2 — Build Docker With Pre Processed Dataset.

First Download the dataset (with coco style txt labelling). 
```bash
# It can also be directly downloaded from the link.
gdown https://drive.google.com/file/d/1IO9fVXa85c3O69aFH-v1xv6pUMfZ8Rfw/view?usp=drive_link --fuzzy
```
### Clone the Repository
---
```bash
git clone https://github.com/mritunjoyh/bosch-assignment.git
cd bosch-assignment
```

### Git LFS Notice
---
This repository contains Git LFS (Large File Storage) files that are required for proper setup.
Ensure these files are available after cloning.

### Download and Prepare the Dataset
---
Create a dataset folder inside the project directory and download the dataset from Google Drive:

### Directory Structure
---
Ensure the project has the following structure before building:

```
boschassignment/
├── Dockerfile
├── models/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```

### Docker Build and run
---
```bash
sudo docker build -t bosch-assign-bdd .

### Then to run
docker run -it bosch_assign_bdd:latest
```


---
# Evaluation Instruction
**All evaluation report are given in [Project Report](report.pdf)**

---
## Dataset Evaluation

Once the Docker container is running, dataset can be evaluated using the provided Streamlit interface.

---

### Steps to Run Evaluation
---
Inside the Docker bash shell, run the following commands:

```bash
cd exploratory-data-analysis
streamlit run main.py
```
### Note:
---
The evaluation process may take some time depending on system performance and dataset size.
Please be patient while the results are being generated.

### Evaluation Results
---
After the Streamlit app completes evaluation, you will see visual outputs and performance metrics. Detailed evlaution is given in [Project Report](report.pdf).

## Model Training and Evaluation

This project involves training and evaluating object detection models on the provided dataset.  
We experimented with **YOLO** and **Faster R-CNN** architectures to identify and localize objects within the dataset.

---

### YOLO Model — Training and Evaluation

#### Training YOLO
---
To train the YOLO model, navigate to the YOLO directory and run the training script:

```bash
cd models/YOLO
python train.py
```
This will start the YOLO training process using the dataset prepared earlier.
Training results, weights, and logs will be saved automatically in the respective output directories.

#### YOLO Evaluation
---
After training, the model is evaluated in two different ways:

Using CLIP and YOLO Combined
Evaluate using CLIP-assisted scoring (semantic + detection):
```python
python eval_clip.py
```

Using Classic YOLO Evaluation
Evaluate using the standard YOLO metric pipeline:
```python
python eval_yolo.py
```

#### Note
---
The YOLO evaluation code uses the default YOLO dataloader.
For custom dataloaders and extended dataset handling, refer to the Faster R-CNN section below.
Detailed descriptions of the evaluation approach are available in the [Project Report](report.pdf).


### Faster R-CNN Model — Training and Evaluation
---
#### Overview
---
The Faster R-CNN model is trained and evaluated using both pretrained and custom lightweight backbones.

We utilized a pretrained model available on Hugging Face:
[Pretrained Faster R-CNN (BDD Fine-tuned)](https://huggingface.co/HugoHE/faster-rcnn-bdd-finetune).

Detailed metrics and comparison results can be found in the evaluation report.

#### Training the Custom Faster R-CNN
---
Navigate to the Faster R-CNN directory to train your model:
```bash
cd models/FasterRCNN
python train.py
```

**Due to GPU and time constraints, a lightweight MobileNet backbone was used instead of ResNet.**
**The full-scale model was not trained completely for this submission.**

**All dataloader and dataset configurations for Faster R-CNN can be found in: models/FasterRCNN/**

### Evaluating Faster R-CNN
---

Two evaluation scripts are available:
```bash
## Evaluation using Pretrained Weights

python eval_frcnn_zoo_trained.py


## Evaluation using Custom-Trained Model

python eval_frcnn.py
```
#### Reports and Results
---
A detailed comparison of both YOLO and Faster R-CNN models,
including CLIP integration, performance graphs, and dataset insights,
is available in the full [Project Report](report.pdf)

# BOSCH Assignment — Setup Instructions

This repository provides the code, environment, and dataset setup required for the BOSCH Assignment.

Follow the instructions below to either **use the pre-built Docker image** or **build it manually**.

---

## Option 1 — Use Pre-Built Docker Image (Recommended)

You can directly download the pre-built Docker image (`.tar.gz`, ~10 GB) from Google Drive and load it into Docker.

```bash
# Download pre-built Docker image (~10GB)
# It can also be directly downloaded from the link.
gdown https://drive.google.com/file/d/19M1Mah8p9qK3SAjuLPAW2YBHoViEoIoz/view?usp=drive_link --fuzzy

# Load the image into Docker
sudo docker load < bosch-assign-bdd.tar.gz

# Verify that the image was loaded
sudo docker images
```
Once loaded, you can run the container directly:
```bash
sudo docker run -it --name temp bosch-assign-bdd bash
```
Note:
The image is large (~10 GB). Ensure you have enough disk space and a stable network connect.

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
git clone <your_repo_link>.git
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
Ensure your project has the following structure before building:

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
---
## Dataset Evaluation
---
Once the Docker container is running, you can evaluate the dataset using the provided Streamlit interface.

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
After the Streamlit app completes evaluation, you will see visual outputs and performance metrics.


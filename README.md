# Garbage classification project with deep learning
This project focuses on building an image classification system that identifies different types of waste (e.g. cardboard, glass, metal, paper, plastic, trash) using deep learning techniques. The goal is to support efficient waste management and recycling efforts by automating the process of sorting garbage through computer vision.

# Features

- Multiclass classification (classifies image into 6 garbage categories)
- Multiple CNN architectures (includes custom CNNs and pretrained MobileNetV2)
- Performance evaluation (confusion matrix, accuracy and loss for all sets)
- Visualization tools (label distribution plots, model architecture summary, prediction on sample images)
- Data augmentation (rescaling and image preprocessing supported)
- Support for external images (predicts on custom images outside training set)

# Dataset
Dataset used: [Garbage classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)

## Installation & Local Run Instructions for project
1. Clone the repository

   ```bash
   git clone https://github.com/your-username/garbage-classification.git
   cd garbage-classification

2. Install dependencies

    ```bash
   pip install -r requirements.txt
   
3. Add your kaggle.json file
   - Go to your [Kaggle account API settings](https://www.kaggle.com/settings)

   - Click on "Create New Token"

   - Please the downloaded kaggle.json file in the root directory of the project

4. The dataset will be downloaded automatically once the notebook is run on your local and kaggle.json is properly added.

## Installation & Local Run Instructions for application

To run the application locally:

1. Go to the GitHub repository and download the ZIP of the project.

2. Unzip the folder to your preferred location.

3. Open a terminal or command prompt.

4. Navigate to the folder where the project was extracted (app folder):

   ```bash
   cd path_to_folder
   
5. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   
6. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

7. Launch the Streamlit app:

   ```bash
   streamlit trash_app.py
   
The app will open in your default web browser at http://localhost:8501.

# Number Recognition using CNN and Pygame

This project is a simple implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits. Users can draw digits on a Pygame interface, and the model will predict the drawn digits. The project includes training the CNN model on the MNIST dataset and using it to recognize digits drawn by the user.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features
- **CNN Model**: A Convolutional Neural Network model is used for digit recognition.
- **Pygame Interface**: A simple drawing interface where users can draw digits.
- **Digit Segmentation**: The drawn image is segmented into individual digits for recognition.
- **Training Script**: The model can be trained on the MNIST dataset.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lnlan1810/Number-Recognition.git
   cd Number-Recognition
   ```

2. Install the required libraries:
   ```bash
   pip install numpy torch torchvision pygame opencv-python pillow
   ```

## Usage
1. **Training the Model**:
   If you want to train the model from scratch, run the script. The model will be saved as `cnn_model.pth`.
   ```bash
   python number_recognition.py
   ```

2. **Using the Pre-trained Model**:
   If a pre-trained model (`cnn_model.pth`) is available, the script will load it automatically.

3. **Drawing and Recognizing Digits**:
   - Run the script:
     ```bash
     python number_recognition.py
     ```
   - A Pygame window will open where you can draw digits.
   - Press `Enter` to recognize the drawn digits.
   - Press `Escape` to clear the drawing area.

## Project Structure
```
number-recognition/
│
├── number_recognition.py       # Main script for training and recognition
├── cnn_model.pth               # Pre-trained model (if available)
├── README.md                   # This file
├── data/                       # Directory for MNIST dataset (automatically downloaded)
└── num.jpg                     # Example image for testing
```


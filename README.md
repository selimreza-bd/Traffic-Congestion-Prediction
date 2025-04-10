# Traffic Congestion Prediction

## Overview
This project predicts traffic congestion using machine learning and AI techniques in Python. Developed as an independent study in transportation engineering, it uses simulated traffic data (vehicle count, speed, hour, weather) to classify road conditions as congested or not congested.

## Features
- **Dataset**: 500 simulated samples with 15% noise to mimic real-world variability.
- **Models**:
  - Random Forest: 88% accuracy.
  - Neural Network: 83% accuracy.
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices.
- **Scenarios**: Tested on realistic cases (e.g., rush hour with rain).

## Results
- Random Forest: Strong congestion detection (96% recall for congested cases, 88% accuracy).
- Neural Network: Robust performance with noisy data (83% accuracy).
- Scenario Predictions:
  - [50, 80, 8, 0] (morning, clear): 15% probability (Not Congested).
  - [150, 30, 17, 1] (evening, rainy): 83% probability (Congested).
  - [80, 60, 2, 0] (late night, clear): 63% probability (Congested).
- Key predictors: Vehicle count and speed (based on feature importance).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/selimreza-bd/Traffic-Congestion-Prediction.git

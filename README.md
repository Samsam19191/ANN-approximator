# Gaussian Density Function Approximation with ANN

This project approximates a 2D Gaussian Probability Density Function (PDF) using an Artificial Neural Network (ANN). 

### Gaussian Parameters:
- **Mean**: [0, 0]
- **Covariance Matrix**: \(\left[\begin{array}{cc} 1 & 1 \\ 1 & 4 \end{array}\right]\)
  
## Project Overview
The ANN was trained to approximate the PDF values, and multiple architectures were experimented with to minimize prediction error.

### Data Generation
Generated 1,000 samples from the specified Gaussian distribution with calculated PDF values used as targets.

## Model Architecture
Primary ANN architecture:
- **Input Layer**: 2 neurons (for \(x_1\) and \(x_2\))
- **Hidden Layers**: 128 neurons, **ReLu** activation
- **Output Layer**: 1 neuron (predicts PDF value)

## Results and Visualizations

### 3D Plot Comparison
Comparison of the true and predicted PDF surfaces.

![Figure_3](https://github.com/user-attachments/assets/5104f11c-37a3-4a79-856e-819b0aaa4e52)


### Contour Plot Comparison
Contour plots of the predicted and true PDFs, illustrating model accuracy across the 2D plane.

![Figure_5](https://github.com/user-attachments/assets/5088f267-8cf4-4c42-aed1-e1a3f3f07dd1)


## Key Learnings
- **Model Stability**: Optimizing the learning rate and using appropriate activations were crucial for convergence.
- **Activation Functions**: ReLU and LeakyReLU performed better in terms of speed and accuracy than Tanh.

# Neural Networks with Backpropagation in Python

**Project**: Building Neural Networks from Scratch  
**Author**: iamtrask  
**Date**: July 12, 2015

---

## Project Overview

This project involves implementing two neural networks from scratch in Python to understand how backpropagation works. The first part focuses on a **simple two-layer neural network** with a single input and output layer, while the second part expands this into a **three-layer neural network** with an additional hidden layer to solve more complex, non-linear problems.

These implementations helped solidify my understanding of how neural networks process input data, adjust weights using backpropagation, and train to minimize error through iterative learning.

---

## Key Concepts and What I Learned

### Part 1: Simple Neural Network

1. **Sigmoid Activation Function**:  
   The network uses the **sigmoid function** to map input values to probabilities between 0 and 1. This is a common choice for binary classification problems, and I learned how activation functions help the network interpret raw input values and generate predictions.

2. **Backpropagation**:  
   Backpropagation is the process of updating the network’s weights by calculating the error at the output and propagating it backward. Through this, I understood how **gradient descent** minimizes errors over time, allowing the network to improve its predictions.

3. **Matrix Multiplication for Weight Updates**:  
   I learned how matrix operations are used to perform forward passes (predictions) and backward passes (error corrections) efficiently, especially when handling multiple training examples in parallel.

4. **Training Through Iterations**:  
   By training the network over thousands of iterations, I saw firsthand how repeating the forward and backward passes gradually reduces the error and improves the model’s accuracy. This iterative process was a great introduction to the concept of **epochs** in machine learning.

### Part 2: 3-Layer Neural Network

1. **Adding a Hidden Layer**:  
   The addition of a hidden layer enabled the network to learn **non-linear patterns**, which a simple two-layer network could not. This helped me understand the importance of **deep learning architectures** for more complex problems.

2. **Multi-Layer Backpropagation**:  
   I learned how to propagate errors back through multiple layers in the network, adjusting the weights between both the input and hidden layers, and the hidden and output layers. This deeper understanding of backpropagation in multi-layer networks is foundational for working with **deep neural networks**.

3. **Non-Linear Pattern Recognition**:  
   The hidden layer allowed the network to recognize combinations of inputs that lead to specific outputs, which is crucial for solving non-linear problems. This concept is central to tasks like **image recognition** and other complex AI challenges.

4. **Weight Initialization for Deep Networks**:  
   Proper weight initialization is critical for networks with multiple layers. I learned how random weight initialization with a mean of zero can help ensure that the network starts in a good place for training, preventing issues like **vanishing gradients**.

---

## Major Features

- **Backpropagation**:  
  Both networks adjust their weights using backpropagation, a key concept in training neural networks. This involves calculating the error at each layer and using that error to adjust the weights in a way that minimizes future errors.

- **Sigmoid Activation Function**:  
  The sigmoid function, used in both implementations, transforms the output of each layer into a probability between 0 and 1, which is essential for making predictions in a binary classification task.

- **Iterative Training**:  
  The networks are trained over thousands of iterations (epochs), allowing them to gradually improve their predictions. This iterative approach helps the network converge on an accurate solution.

---

## Technical Challenges Overcome

1. **Understanding Backpropagation**:  
   The most significant challenge was fully understanding how backpropagation works, especially in a multi-layer network. Breaking down the process step by step allowed me to grasp how errors propagate back through the network and adjust weights at each layer.

2. **Matrix Operations for Forward and Backward Passes**:  
   Implementing matrix operations for both the forward and backward passes was key to making the neural networks work efficiently. This involved understanding how to multiply inputs by weights and adjust those weights based on the output errors.

3. **Training Multi-Layer Networks**:  
   Expanding the network to three layers introduced additional complexity in terms of weight initialization, backpropagation, and error handling. Learning how to manage these layers helped me understand the foundations of **deep learning**.

---

## What I Learned

This project was an invaluable exercise in understanding the fundamentals of neural networks and machine learning. I gained practical knowledge of how **activation functions**, **backpropagation**, and **weight adjustments** work together to make neural networks learn from data. Additionally, the introduction of a hidden layer showed me how more complex architectures can solve non-linear problems and improve the accuracy of machine learning models.

---

## Conclusion

By building these neural networks from scratch, I developed a solid foundation in machine learning, backpropagation, and deep learning architectures. These concepts are essential for understanding and developing more sophisticated AI models in the future.


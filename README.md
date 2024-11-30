### **Python Mini-Project: Classification Using Convolutional Neural Networks**

#### **Project Objective**  
This Python mini-project aims to classify flower species using Convolutional Neural Networks (CNNs). The Oxford Flowers 102 dataset was utilized to implement a deep learning pipeline that includes preprocessing, model training, evaluation, and visualization using Python.

#### **Dataset Details**  
The Oxford Flowers 102 dataset comprises a variety of flower images with the following division:  
- **Total Images**: ~8,189  
- **Training Dataset**: ~6,149 images (75%)  
- **Validation Dataset**: ~1,020 images (12.5%)  
- **Testing Dataset**: ~1,020 images (12.5%)  

All images were resized to 224 x 224 pixels to ensure compatibility with the MobileNetV2 architecture.

#### **Python Libraries Used**
- **Data Preprocessing and Augmentation**: NumPy, TensorFlow, Keras  
- **Visualization**: Matplotlib, Seaborn  
- **Evaluation**: Scikit-learn for precision, recall, and F1 score  

#### **Workflow**
1. **Preprocessing**:  
   - Images resized to 224 x 224 pixels.  
   - Data augmentation techniques applied: random flipping, brightness and contrast adjustments, and cropping.  

2. **Model Architecture**:  
   - Base Model: MobileNetV2 pre-trained on ImageNet (excluding final classification layers).  
   - Additional Layers:  
     - GlobalAveragePooling2D for dimensionality reduction.  
     - Fully connected Dense layer (512 neurons, ReLU activation).  
     - Dropout layer for overfitting mitigation.  
     - Final Dense output layer with 102 units (softmax activation).  

3. **Training and Optimization**:  
   - **Loss Function**: Sparse Categorical Crossentropy  
   - **Optimizer**: Adam with an adaptive learning rate (0.001)  
   - **Batch Size**: 16  
   - **Epochs**: 30 with learning rate decay and early stopping.  

4. **Evaluation**:  
   - Accuracy: Training (97.93%), Validation (94.42%), Testing (94.63%).  
   - Metrics: Precision (0.95), Recall (0.94), F1 Score (0.94).  

5. **Visualization and Interpretation**:  
   - Training and validation accuracy/loss curves.  
   - Test predictions visualized with Matplotlib.  

#### **Key Takeaways**
This project demonstrates Python's power in building and training deep learning models efficiently.
Leveraging MobileNetV2's transfer learning capability enabled high classification accuracy even with a relatively small dataset.

---

This write-up reflects the project as a Python-based deep learning application, emphasizing the tools and workflow typical of Python projects. 

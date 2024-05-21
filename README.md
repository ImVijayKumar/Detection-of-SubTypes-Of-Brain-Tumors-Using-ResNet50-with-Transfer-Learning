# Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning
## INTRODUCTION
Meningiomas, gliomas, and pituitary tumors are examples of abnormal growths that affect the central nervous system and can be classified as either benign or malignant (benign/malignant).MRI and CT scans are used in diagnostic procedures to produce detailed pictures for preliminary assessment. New developments in technology have enhanced contrast and resolution in imaging, which helps with diagnosis accuracy. Computer-aided diagnosis (CAD) systems use Convolutional Neural Networks (CNN) incorporated with medical imaging. Modern medical imaging technology like CNN is excellent at diagnosing diseases without requiring a lot of preprocessing, especially for CT and MRI images. Input, convolution, RELU, fully connected, classification, and output layers are all part of the CNN architecture. Convolution and downsampling are two essential CNN operations that allow for efficient raw picture processing.CNN is used by machine learning systems to categorize brain MRI pictures as normal or abnormal and to grade aberrant images for various types of brain tumors. In our experiment, we classified brain tumors using the ResNet-50 architecture. used pre-trained ResNet-50 and applied transfer learning techniques to improve model performance. The model was trained using T1-weighted, contrast-enhanced MRI images of brain tumors. Because of its intrinsic nature, the CNN model proved effective even with significant preprocessing. The main goal is to raise the accuracy of brain tumor grading so that doctors may better plan treatments and see higher recovery rates. presents a new CNN architecture that departs from popular transfer learning methods.
For computer vision applications, CNN combined with Transfer Learning, Python, and PyTorch are among the technologies used.

## METHODOLOGY
### 1. Dataset
The brain tumor dataset from Cheng, Jun, which can be accessed for free online at https://figshare.com/articles/brain_tumor_dataset/1512427/5, was utilized in this work. The collection comprises 3064 T1 weighted and contrast-enhanced brain MRI images, classified into three categories: pituitary tumor, meningioma, and glioma. The number of photos in the dataset for each class is listed in Figure 1. The patient ID (PID), tumor mask, tumor border, and class label are all fully described and provided for every image in the dataset. The lesion mask, which is used to crop the tumor region of interest (ROI), is the most crucial piece of information after the class label.

Figure 1: Summary of Used Image Dataset

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/347db50f-c3bb-48ce-9e41-63d63f69a676)

### 2.Convolutional Neural Network (CNN)
The most popular deep-feed forward neural network at the moment is the convolutional neural network (CNN), which can handle both 1D and 2D visual data inputs. The input layer, convolution layer, RELU layer, fully connected layer, classification layer, and output layer are the many layers that make up a CNN in general. Convolution employing a trainable filter with a predetermined size and weights that are
modified throughout the downsampling process in the training phase to obtain a high accuracy are the two main processes that CNN is founded on. The photos of both cropped and uncropped brain tumors are kept in a database for this study, and three files are made, one for each kind of tumor: glioma, meningioma, and pituitary.
### 3.Proposed CNN Architecture
•	Architecture: Utilized ResNet-50, a deep residual network architecture.
•	Layer Modification: Adapted the fully connected layer for our specific brain tumor classification task with 4 output classes (None, Meningioma, Glioma, Pituitary).
•	Transfer Learning: Exploited the pretrained ResNet-50 model for feature extraction and pattern recognition.
•	Fine-tuning: Retrained the model on our brain tumor dataset to enhance its predictive capabilities.
•	Freezing Convolutional Layers: Initially, convolutional layers are frozen to retain generic features.
•	Adaptive Learning Rates: Trained the model with variable learning rates for optimal convergence.
•	Loss Function: Employed Logarithmic Sigmoid loss suitable for multiclass classification.
•	Activation Function: Used LogSigmoid in the final layer for probability distribution across classes.
### 4. Performance Evaluation
Based on these produced confusion matrices, a comparison between the CNN architecture outputs and the original picture label for each was done in order to assess the effectiveness of the suggested CNN architecture. In general, we may compute the accuracy, sensitivity, precision, and specificity to gauge how exactly the brain tumor is being evaluated using these created confusion matrices. To assess the effectiveness of the proposed classification method, four statistical indices—true positive (TP), false positive (FP), false negative (FN), and true negative (TN)—are computed from the confusion matrix that is produced.
### 5.System Implementation
1.	Preprocessing: To guarantee consistency and improve model performance, MRI images are pre-processed using methods including scaling and normalization.
2.	Feature Extraction: ResNet-50 automatically extracts complex characteristics from MRI data by using transfer learning, which enables reliable brain tumor classification.
3.	Model Training: By fine-tuning the parameters of the pre-trained ResNet-50 model, particularly for brain tumor classification tasks, its accuracy, and effectiveness are increased.
4.	Model Evaluation: The effectiveness and dependability of the trained ResNet-50 model are validated by a thorough evaluation that makes use of performance indicators including accuracy and sensitivity.
 5.	Integration and Deployment: The ResNet-50 model may be more easily used in actual healthcare set- tings by being seamlessly integrated into an intuitive clinical system.
6.	Testing and Validation: Robust testing and validation procedures guarantee the ResNet-50 model's accuracy and resilience across a variety of datasets and clinical settings.
7.	Continuous Improvement: The ResNet-50 model's continuous optimization and efficacy are ensured by ongoing improvements and refinements made in response to user input and developments in machine learning techniques.

Figure 2: Proposed System Architecture

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/59df1ab8-d81b-4a02-80e2-d0c4ee3fb4f7)

### 6.Results
### 6.1	Validation Results
Validation of our system involves rigorous testing against diverse datasets and clinical scenarios to ensure its reliability, accuracy, and generalizability in real-world healthcare settings.

Figure 1: Confusion Matrix for Model's performance

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/eaf15b44-9f78-41e2-bbd5-94b3fa0f8b0d)

Figure 2: Graphical Representation of the Model’s Accuracy for Training and Validation Data

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/fe5a733b-e2cc-4e45-bc2d-4fc125ff8597)

Figure 3: Graphical Representation of the Model’s Loss for Training and Validation Data

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/07fcadde-1106-4013-ac3a-fecd4f8b787b)

### 6.2	Final Prediction Outputs
Here are our final predictions for the classes pituitary tumor, Meningioma, and Glioma

Figure 1: Home Page

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/86eaa313-fb23-4dbe-937a-f8c05b04bde1)

Figure 2: Prediction for Glioma

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/38d61232-4746-4de0-97f1-ca5b81fd72ac)

Figure 3: Prediction for Meningioma

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/53e5a3aa-4102-4ce4-b328-9cff38ec7a81)

Figure 4: Prediction for Pituitary

![image](https://github.com/ImVijayKumar/Detection-of-SubTypes-Of-Brain-Tumors-Using-ResNet50-with-Transfer-Learning/assets/142383380/cd09cead-d455-4ab6-8d15-03ffea351436)

## Conclusion
Our initiative, NeuralBlack, offers a potent remedy for the identification of brain cancers through the categorization of medical images. We have obtained impressive results by combining state-of-the-art technology with a well-trained ResNet-50 model. Let's review the main advantages:
•	Accurate Tumor Classification
•	User-Friendly Interface
•	Fast and Efficient
•	Scalability and Adaptability
•	Open-Source Contribution
•	Impact on Healthcare
In summary, NeuralBlack is a useful tool that advances healthcare by showcasing the potential of contemporary machine learning. It also plays a significant role in the medical field.



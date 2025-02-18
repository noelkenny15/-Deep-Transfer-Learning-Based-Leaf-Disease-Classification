# -Deep-Transfer-Learning-Based-Leaf-Disease-Classification
This project is a deep transfer learning-based leaf disease detection system that utilizes transfer learning to identify and classify different types of leaf diseases. Trained on a vast dataset of images, the model is designed to assist agricultural professionals and enthusiasts in quickly and accurately diagnosing plant diseases.
## Domain
The project domain is related to deep learning technology, by utilizing deep learning algorithms to predict plant illnesses. The technique we utilize in this case is the
Convolutional Neural Network [CNN] Algorithm are VGG16, ResNet, Inception v2,
Mobile Net for Paddy Disease Prediction
## Usage:
To use the model for leaf disease detection, follow these steps:
- Make sure you have a Python environment set up with the necessary libraries installed. You can use the provided requirements.txt file to set up the required dependencies.
  pip install -r requirements.txt
- Run main.py:
  streamlit run main.py
## Model Details:
The leaf disease detection model employs deep learning techniques and utilizes transfer learning to leverage the pre-trained knowledge of a base model. It has been trained on a dataset featuring images of 33 different leaf diseases. For details on the architecture, dataset, and training process, please refer to the provided code and documentation.
## Algorithm:
- Step1 : Start
- Step2 : Using CNN models requires selecting a dataset first. After that, the dataset needs to be prepared for training.
- Step3 : Create the training data and assign features and labels ,normalize and convert labels into categorical data.
- Step4 : Split x and y for using in CNN models.
- Step5 : Give the directories of train and validation datasets for the training model.
- Step6 : Save the model for testing.
- Step7 : Test the model using the test dataset.
- Step8 : It gives the output as affected disease name.
- Step9 : Stop

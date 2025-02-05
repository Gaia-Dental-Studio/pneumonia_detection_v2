# Pneumonia Detection

This repository consists of the training and helper code to deploy the pneumonia model.

The model was trained to classify X-ray images into three categories: Pneumonia TBC, Pneumonia non-TBC, and Normal. The datasets used for the training/fine-tuning are the [Pneumonia chest X-ray images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and [Tuberculosis chest X-ray images](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) downloaded from Kaggle. The data augmentation and class-weighted approach were implemented to handle the imbalanced data distribution for each category. The model was trained on three different deep learning architectures: CNN and two pre-trained models (VGG and Mobilenet) model.

The model accessed link and accuracy are listed below:
1. [CNN](https://drive.google.com/file/d/1--ThJOloeL444-PCdbQzsEkltjkp9zGD/view?usp=drive_link): 99.1%
2. [VGG](https://drive.google.com/file/d/1QzTzD5CAAdI6jc4Wye4FNXVYJgw4f2Nv/view?usp=drive_link):99.5%
3. [MobilNet](https://drive.google.com/file/d/1bDl7EsE5Cy_YwrMJvWoHGU03u2jYIYEf/view?usp=drive_link): 99.7%

Two helper codes, api.py (Flask backend) and app.py (Streamlit frontend), are provided to run the model demo. In this helper code (api.py), the upload file (x-ray image) is set to remove automatically after the process. Modifications are needed to store the file.

_Note_: 
The datasets used in the training process mainly contain the child's X-ray image. Therefore, the accuracy of the results might not be relevant for detecting pneumonia in adult X-rays.

## Requirements

```
Python 3.x
PIL
flask
io
numpy
os
requests
streamlit
tensorflow
```

## Installation
1. Clone this repository and the models on the link above.
    ```
    git clone https://github.com/Gaia-Dental-Studio/pneumonia_detection_v2.git
    ```
2. Navigate to pneumonia_detection_v2 directory, create 'model' directory and move the downloaded model into it.
3. Install all requirements dependencies.
4. Open the api.py and check the model path and input size for the model are matched. For example:
    ```
    model = tf.keras.models.load_model('model/x-ray_cnn_model.h5')
    img = image.load_img(img_path, target_size=(224, 224))  
    ```
    or
    ```
    model = tf.keras.models.load_model('model/x-ray_vgg_model.h5')
    img = image.load_img(img_path, target_size=(244, 244))  
    ```
5. Open two terminals or command prompts and run the following syntax in parallel.
    ```
    python api.py
    ```
    and
    ```
    streamlit run app.py
    ```
6. Open your web browser and navigate to the provided URL from the streamlit.
7. Upload X-ray images, and the result will automatically appear below the image.

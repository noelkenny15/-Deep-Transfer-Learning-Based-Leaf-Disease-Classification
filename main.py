import os
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import CNN

# Load CSV Files
disease_info = pd.read_csv(
    'C:/Users/noeL/Downloads/Plant-Disease-Detection-main/Flask Deployed App/disease_info.csv',
    encoding='cp1252'
)
supplement_info = pd.read_csv(
    'C:/Users/noel/Downloads/Plant-Disease-Detection-main/Flask Deployed App/supplement_info.csv',
    encoding='cp1252'
)

# Load Model
model = CNN.CNN(39)
model.load_state_dict(torch.load(
    "D:/MINOR/Plant-Disease-Detection-main/Flask Deployed App/plant_disease_model_latest.pt"
))
model.eval()

# Prediction Function
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Initialize Flask App
app = Flask(__name__, template_folder='D:/MINOR/Plant-Disease-Detection-main/Flask Deployed App/templates')

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def aienginepage():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        filepath = os.path.join(
            'C:/Users/lakshmidhar/Downloads/Plant-Disease-Detection-main/Flask Deployed App/static/uploads',
            filename
        )
        image.save(filepath)
        pred = prediction(filepath)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement_name'][pred]
        supplement_image_url = supplement_info['supplement_image'][pred]
        supplement_buy_link = supplement_info['buy_link'][pred]
        return render_template(
            'submit.html', title=title, desc=description, prevent=prevent,
            image_url=image_url, pred=pred, sname=supplement_name,
            simage=supplement_image_url, buy_link=supplement_buy_link
        )

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement_image']),
        supplement_name=list(supplement_info['supplement_name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy_link'])
    )

if __name__ == '__main__':
    app.run(debug=True)

import torch.nn as nn

# Define CNN Model
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

# Dictionary Mapping Indices to Class Names
idx_to_classes = {
    0: 'Apple Apple scab',
    1: 'Apple Black rot',
    2: 'Apple Cedar apple rust',
    3: 'Apple healthy',
    4: 'Background without leaves',
    5: 'Blueberry healthy',
    6: 'Cherry Powdery mildew',
    7: 'Cherry healthy',
    8: 'Corn Cercospora leaf spot Gray leaf spot',
    9: 'Corn Common rust',
    10: 'Corn Northern Leaf Blight',
    11: 'Corn healthy',
    12: 'Grape Black rot',
    13: 'Grape Esca (Black Measles)',
    14: 'Grape Leaf blight (Isariopsis Leaf Spot)',
    15: 'Grape healthy',
    16: 'Orange Haunglongbing (Citrus greening)',
    17: 'Peach Bacterial spot',
    18: 'Peach healthy',
    19: 'Pepper, bell Bacterial spot',
    20: 'Pepper, bell healthy',
    21: 'Potato Early blight',
    22: 'Potato Late blight',
    23: 'Potato healthy',
    24: 'Raspberry healthy',
    25: 'Soybean healthy',
    26: 'Squash Powdery mildew',
    27: 'Strawberry Leaf scorch',
    28: 'Strawberry healthy',
    29: 'Tomato Bacterial spot',
}

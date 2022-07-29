# Melanoma-Classifier
## Web app to classify melanoma using federated learning

The cells that make melanin, the pigment responsible for your skin's color, can grow into melanoma, the most dangerous kind of skin cancer. Melanoma can also develop in your eyes and, very rarely, within your body, including in your throat or nose.

It is essential to track the progress of a skin legion to be able detect signs of a positive melanoma case early on.

In this project, we designed an app that allows users to classify and visualize their skin legion image data.

With the help of federated learning, the app also functions as a way to enhance the base model's parameters. Users with similar datasets can insert their labeled data into the app where it will retrain the model with new parameters. This is based on the federated learning architecture where instead of giving raw data to a central server, each client gets the model on their own server where they can train on their own data and send back only the parameters they got from the training. So each client only sends back these parameters to the central server where all the parameters are aggregated, and a new model is formed and can be used on new data.


![DSA 5 Poster](https://user-images.githubusercontent.com/54022220/181771710-c2ba1bc0-3707-4483-ac84-9dff3e7e2ce2.png)

## How to run?

### Download necessary data before running the app

- [isicdata plain train/test images](https://www.kaggle.com/datasets/nroman/melanoma-external-malignant-256/download?datasetVersionNumber=1)

- [isicdata tfrecords data](https://www.kaggle.com/datasets/cdeotte/melanoma-512x512/download?datasetVersionNumber=3)

- The data should be placed within the project path as the structure below;

```bash
isicdata
├── datasets
│   ├── doctor_case.csv
│   ├── doctor_case2.csv
│   ├── test.csv
│   ├── train.csv
│   └── train_concat.csv
├── test
├── tfrecords
└── train
```

### To run the app, type the following commands into the terminal:


`git clone git@github.com:natfal14/Melanoma-Classifier.git`

`cd Melanoma-Classifier`

`pip3 install -r requirements.txt`

`streamlit run app_v2.py`


The Streamlit app will start running in http://localhost:8501

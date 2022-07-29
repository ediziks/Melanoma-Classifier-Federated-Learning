import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
import warnings

# from tenacity import TryAgain
from tensorflow.keras import layers as L
# from tensorflow.keras.applications.efficientnet import EfficientNetB2 as efn
import efficientnet.tfkeras as efn
import tensorflow as tf
# from collections import Counter
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tqdm.notebook import tqdm
# import sklearn
import os
import flwr as fl
import glob
from datetime import datetime
import argparse
warnings.filterwarnings('ignore')


# define device
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print('Number of cores used: ', strategy.num_replicas_in_sync)
print('Number of devices: ', strategy.num_replicas_in_sync)

# Feature Eng
def train_feature_eng(df):
    df['path'] = './isicdata/train/train/' + df['image_name'] + '.jpg'
    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('torso')
    df['sex'] = df['sex'].fillna('male')
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
    
    return df


# def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
#                lr_min=0.000001, lr_rampup_epochs=20, 
#                lr_sustain_epochs=0, lr_exp_decay=.8):
#     lr_max = lr_max * strategy.num_replicas_in_sync

#     def lrfn(epoch):
#         if epoch < lr_rampup_epochs:
#             lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
#         elif epoch < lr_rampup_epochs + lr_sustain_epochs:
#             lr = lr_max
#         else:
#             lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
#         return lr
    
#     return lrfn

# def test_feature_eng(df):
#     df['path'] = './isicdata/test/test/' + df['image_name'] + '.jpg'
#     df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('torso')
#     df['sex'] = df['sex'].fillna('male')
#     df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
    
#     return df


# Model Implemention

# def load_model(weights='imagenet', metrics=['accuracy']):    

#     # Define the model as efficientnet-b2
#     model = tf.keras.applications.efficientnet.EfficientNetB2(
#         include_top=False,
#         weights=weights,
#         input_shape=(224, 224, 3),
#         pooling='avg',
#         classes=2,
#         classifier_activation='softmax'
#     )

#     # Compile the model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(lr=0.0001),
#         loss='sparse_categorical_crossentropy',
#         metrics=metrics
#     )
    
#     return model

# def get_latest_weights(path='./workspace/checkpoints/*'):
#     # get the latest model weights from the path
#     list_of_files = glob.glob(path)
#     return max(list_of_files, key=os.path.getctime)


# model = load_model(weights=get_latest_weights())


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

def load_model():
    IMAGE_SIZE = [384, 384]
    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB2(
                input_shape=(*IMAGE_SIZE, 3),
                weights='imagenet',
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1024, activation = 'relu'), 
            L.Dropout(0.3), 
            L.Dense(512, activation= 'relu'), 
            L.Dropout(0.2), 
            L.Dense(256, activation='relu'), 
            L.Dropout(0.2), 
            L.Dense(128, activation='relu'), 
            L.Dropout(0.1), 
            L.Dense(1, activation='sigmoid')
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

        model.compile(
            optimizer=opt,
            # loss = 'binary_crossentropy',
            loss=focal_loss(gamma=2., alpha=.25),
            metrics=['binary_crossentropy', 'accuracy'],
        )
    
    return model

# train prep
def prepare_train_df(df, mela_count=40, bening_count=8, train_ratio=0.8, batch_size=1):
    df = train_feature_eng(df)
    # limit max counts
    mela_count = 584 if mela_count > 584 else mela_count
    bening_count = 32542 if bening_count > 32542 else bening_count
    # separate 0 and 1 targets
    df_mela = df[df['target'] == 1]
    df_benign = df[df['target'] == 0]
    # shuffle the data
    df_mela = df_mela.sample(frac=1)
    df_benign = df_benign.sample(frac=1)
    # select amounts
    df_mela = df_mela[:mela_count]
    df_benign = df_benign[:bening_count]
    # join data
    df2 = pd.concat([df_mela, df_benign], ignore_index=True)
    # shuffle the final data
    df2 = df2.sample(frac=1)
    # train set ratio
    train_ratio = train_ratio
    slice_num = int(len(df2)*(train_ratio))

    # prepare tf data - with image files from path
    train_dataset = tf.data.Dataset.from_tensor_slices((df2['path'][0:slice_num].values, df2['target'][0:slice_num].values))
    test_dataset = tf.data.Dataset.from_tensor_slices((df2['path'][slice_num:-1].values, df2['target'][slice_num:-1].values))

    train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x)), (384, 384)), y))
    test_dataset = test_dataset.map(lambda x, y: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x)), (384, 384)), y))

    # combine into batches
    batch_size = batch_size
    train_dataset = train_dataset.shuffle(buffer_size=batch_size).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset, batch_size, slice_num


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, batch_size, slice_num):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.slice_num = slice_num

    def get_parameters(self):
        return Exception("Not implemented (server-side parameter initialization)")
    
    def fit(self, parameters, config):
        # df = pd.read_csv('./isicdata/datasets/train.csv')
        # train_dataset, test_dataset, batch_size, slice_num = prepare_train_df(df)
        self.model.set_weights(parameters)
        self.model.fit(x=self.train_dataset,
            epochs=1,
            validation_steps=self.slice_num//self.batch_size,
            validation_data=self.test_dataset,
            class_weight = {0:0.025, 1: 11}
        )
        _, _, acc = self.model.evaluate(self.test_dataset)
        return self.model.get_weights(), len(self.train_dataset), {'accuracy': acc}
    
    def evaluate(self, parameters, config):
        # df = pd.read_csv('./isicdata/datasets/train.csv')
        # train_dataset, test_dataset, _, _ = prepare_train_df(df)
        train_dataset, test_dataset = self.train_dataset, self.test_dataset
        self.model.set_weights(parameters)
        loss, _, accuracy = self.model.evaluate(test_dataset)
        return loss, len(train_dataset), {"accuracy": accuracy}



def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--path", type=str, default="./isicdata/datasets/doctor_case2.csv", help="Path to the dataset")
    args = parser.parse_args()

    df = pd.read_csv(args.path)
    train_dataset, test_dataset, batch_size, slice_num = prepare_train_df(df, mela_count=6, bening_count=43, train_ratio=0.8)

    model = load_model()

    # Start Flower client
    client = FlowerClient(model, train_dataset, test_dataset, batch_size, slice_num)

    fl.client.start_numpy_client(
        server_address="[::]:8080",
        client=client,
    )


if __name__ == "__main__":
    main()
from secrets import choice
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
from PIL import Image
import matplotlib.pyplot as plt
import io
import subprocess
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense
import efficientnet.tfkeras as efn
from tensorflow.keras import layers as L
import glob
import os
from PIL import Image
import contextlib
import time


def main_page():
    st.markdown("# Image Detection")

def page2():
    st.markdown("# Data Exploration")

def page3():
    st.markdown("# Training")

page_names_to_funcs = {
    "Image Detection": main_page,
    "Data Exploration": page2,
    "Training": page3,
}

st.sidebar.image("logo.jpeg", use_column_width=True)
st.sidebar.title("Skin Cancer Detection through Neural Network on Federated Learning")

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


if selected_page == "Image Detection":
    st.write("The cells that make melanin, the pigment responsible for your skin's color, can grow into melanoma, the most dangerous kind of skin cancer. Melanoma can also develop in your eyes and, very rarely, within your body, including in your throat or nose.")
    st.write("Here you can insert your own skin legion image and have the app predict how likely it is to be a postitive case of melanoma.")
    st.write("You can select one of the models below.")
    choice = "Image"
    if choice == "Image":
        files = os.listdir('workspace/clientResults')
        weight_files = []
        for i in files:
            if 'round' in i or 'base' in i:
                weight_files.append(i)

        selected_weight = st.selectbox("Select model to predict", weight_files)
        model = tf.keras.Sequential([
                efn.EfficientNetB2(
                    input_shape=(*[256, 256], 3),
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

        if selected_weight is not None:
            model.load_weights('./workspace/clientResults/' + selected_weight)
            st.info('Loaded model weights: ' + selected_weight)

        image_file = st.file_uploader("Upload Image", type=["jpg"])

        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, width=300)
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = tf.cast(image, tf.float32) / 255.0
            img_size = [256, 256]
            image = tf.reshape(image, [*img_size, 3])
            image = tf.expand_dims(image, axis=0)

            pred = model.predict(image)
            pred_per = round(pred[0][0] * 100, 2)
            # st.info(pred)
            # st.write('Prediction:  %', pred_per, 'melanoma risk')
            if pred_per > 50:
                # st.markdown(f'<*font color=‘red’>Melanoma risk is high by %{pred_per}</*font>', unsafe_allow_html=True)
                original_title = f'<p style="color:Red;">Melanoma risk is high by %{pred_per}</p>'
                st.markdown(original_title, unsafe_allow_html=True)
            else:
                original_title = f'<p style="color:Green;">Melanoma risk is low by %{pred_per}</p>'
                st.markdown(original_title, unsafe_allow_html=True)

elif selected_page == "Data Exploration":
    st.subheader("Data Exploration")
    st.write("Here you can see your the specification of the data")


    # st.subheader("Melanoma case")

    # image1 = Image.open("isicdata/train/train/ISIC_0351666.jpg")
    # image2 = Image.open("isicdata/train/train/ISIC_0369831.jpg")
    # image3 = Image.open("isicdata/train/train/ISIC_0489267.jpg")


    # col1, col2, col3  = st.columns([2,2,2])

    # with col1:
    #     st.image(image1, width=230, use_column_width='never', caption='head/neck')
    # with col2:
    #     st.image(image2, width=230, use_column_width='never', caption='upper extremity')
    # with col3:
    #     st.image(image3, width=230, use_column_width='never', caption='lower extremity')


    # st.subheader("Non-melanoma case")

    # image1 = Image.open("isicdata/train/train/ISIC_2637011.jpg")
    # image2 = Image.open("isicdata/train/train/ISIC_0015719.jpg")
    # image3 = Image.open("isicdata/train/train/ISIC_0052212.jpg")


    # col1, col2, col3  = st.columns([2,2,2])

    # with col1:
    #     st.image(image1, width=230, use_column_width='never', caption='head/neck')
    # with col2:
    #     st.image(image2, width=230, use_column_width='never', caption='upper extremity')
    # with col3:
    #     st.image(image3, width=230, use_column_width='never', caption='lower extremity')

    # df = pd.read_csv('isicdata/train_concat.csv')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        diag = df.diagnosis.value_counts()
        st.subheader("Your dataset")
        st.write(df)

        st.subheader("Vizualization of your dataset")

        orange_black = [
        '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
        ]


        st.write("Sunburst Chart Benign/Malignant > Sex > Location")
        df1 = df.dropna()
        fig = px.sunburst(df1, 
                        path=['benign_malignant', 'sex', 'location'],
                        color='sex',
                        color_discrete_sequence=orange_black,
                        maxdepth=-1,
                        )

        fig.update_traces(textinfo='label+percent parent')
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


        
        fig = px.pie(diag,
                    values='diagnosis',
                    names=diag.index,
                    color_discrete_sequence=orange_black,
                    hole=.4,
                    title='Diognosis')
        fig.update_traces(textinfo='percent+label', pull=0.05)

        st.plotly_chart(fig, use_container_width=True)


        cntstr = df.location.value_counts().rename_axis('location').reset_index(
        name='count')

        fig = px.treemap(cntstr,
                    path=['location'],
                    values='count',
                    color='count',
                    color_continuous_scale = orange_black,
                    title='Scans by Location')

        fig.update_traces(textinfo='label+percent entry')
        st.plotly_chart(fig, use_container_width=True)



        fig = plt.figure(constrained_layout=True, figsize=(10, 6))

        # Creating a grid:

        grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

        ax1 = fig.add_subplot(grid[0, :2])

        # Set the title.

        ax1.set_title('Gender Distribution')

        # ax1.set_xticks(range(2))

        # ax1.set_xticklabels(['male', 'female'])
        # ax1.set(xticks=range(len(df)), xticklabels=['male', 'female'])




        chart = sns.countplot(df.sex.sort_values(ignore_index=True),
                    alpha=0.9,
                    ax=ax1,
                    color='#fdc029',
                    x='sex'
                    )
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

        ax1.legend()

        # Customizing the second grid.

        ax2 = fig.add_subplot(grid[0, 2:])

    
        # Plot the countplot.

        sns.countplot(df.location,
                    alpha=0.9,
                    ax=ax2,
                    color='#fdc029',
                    order=df['location'].value_counts().index)
        ax2.set_title('Anatom Site Distribution')

        ax2.legend()

        ax3 = fig.add_subplot(grid[1, :])

        # Set the title.

        ax3.set_title('Age Distribution')

        # Plot the histogram.
        df_age = df.dropna()
        sns.distplot(df_age.age, ax=ax3, color='#fdc029')

        ax3.legend()

#         ax4 = fig.add_subplot(grid[1, :])

# # Set the title.

#         ax4.set_title('Age Distribution by Gender')
#         df_age1 = df.dropna()
#         # Plot

#         sns.distplot(df_age1[df_age1.sex == 'female'].age,
#                     ax=ax3,
#                     label='Female',
#                     color='#fdc029')
#         sns.distplot(df_age1[df_age1.sex == 'male'].age,
#                     ax=ax3,
#                     label='Male',
#                     color='#171820')
#         ax4.legend()


        st.plotly_chart(fig, use_container_width=True)

        # st.plotly_chart(fig)

        # cntstr = df.location.value_counts().rename_axis('location').reset_index(
        # name='count')

        # fig = px.treemap(cntstr,
        #             path=['location'],
        #             values='count',
        #             color='count',
        #             color_continuous_scale = orange_black,
        #             title='Scans by Location')

        # fig.update_traces(textinfo='label+percent entry')
        # st.plotly_chart(fig, use_container_width=True)


        # Creating a customized chart and giving in figsize etc.
      

        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        # Creating a grid
        grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

        # Customizing the first grid.

        ax1 = fig.add_subplot(grid[1, :2])
        # Set the title.
        ax1.set_title('Scanned Body Parts - Female')

        # Plot:

        sns.countplot(
            df[df['sex'] == 'female'].location.sort_values(ignore_index=True),
            alpha=0.9,
            ax=ax1,
            color='#fdc029',
            label='Female',
            order=df['location'].value_counts().index)
        ax1.legend()

        # Customizing the second grid.
        ax2 = fig.add_subplot(grid[1, 2:])

        # Set the title.

        ax2.set_title('Scanned Body Parts - Male')

        # Plot.

        sns.countplot(
            df[df['sex'] == 'male'].location.sort_values(ignore_index=True),
            alpha=0.9,
            ax=ax2,
            color='#171820',
            label='Male',
            order=df['location'].value_counts().index)

        ax2.legend()

        # Customizing the third grid.

        ax3 = fig.add_subplot(grid[0, :])

        # Set the title.

        ax3.set_title('Malignant Ratio Per Body Part')

        # Plot.

        loc_freq = df.groupby('location')['target'].mean().sort_values(
            ascending=False)
        sns.barplot(x=loc_freq.index, y=loc_freq, palette=orange_black, ax=ax3)

        ax3.legend()

        # plt.show()
        st.plotly_chart(fig, use_container_width=True)


  
        # fig = px.pie(diag,
        #             values='diagnosis',
        #             names=diag.index,
        #             color_discrete_sequence=orange_black,
        #             hole=.4,
        #             title='Diognosis')
        # fig.update_traces(textinfo='percent+label', pull=0.05)

        # st.plotly_chart(fig, use_container_width=True)


        # st.write("Sunburst Chart Benign/Malignant > Sex > Location")
        # df1 = df.dropna()
        # fig = px.sunburst(df1, 
        #                 path=['benign_malignant', 'sex', 'location'],
        #                 color='sex',
        #                 color_discrete_sequence=orange_black,
        #                 maxdepth=-1,
        #                 title='Sunburst Chart Benign/Malignant > Sex > Location')

        # fig.update_traces(textinfo='label+percent parent')
        # fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        # st.plotly_chart(fig, use_container_width=True)
elif selected_page == "Training":

    test_df = st.file_uploader("Choose a file", key='diagnosis')
    if test_df is not None:
        def run_command(args):
            st.info(f"Federated Learning network is now training with 1 server and 2 clients")
            result = subprocess.run(args, capture_output=True, text=True)
            try:
                result.check_returncode()
                st.info(result.stdout)
            except subprocess.CalledProcessError as e:
                st.error(result.stderr)
                raise e

        if st.button("Train"):
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    with contextlib.redirect_stderr(stderr):
                        with st.spinner(f'Training with {test_df.name}...'):
                            run_command(["bash", "run.sh", "-p", f"isicdata/datasets/{test_df.name}"])
            except Exception as e:
                st.write(f"Failure while executing: {e}")
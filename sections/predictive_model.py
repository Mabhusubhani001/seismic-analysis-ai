import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Add, Concatenate
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create balanced bins
def create_balanced_bins(series, n_bins=4):
    non_null = series.dropna()
    bins = pd.qcut(non_null, q=n_bins, duplicates='drop')
    labels = [f'Level_{i+1}' for i in range(len(bins.unique()))]
    binned = pd.qcut(series, q=n_bins, labels=labels, duplicates='drop')
    return binned

# Function to preprocess data
def preprocess_data(df):
    df['cdi_class'] = create_balanced_bins(df['cdi'])
    df['mmi_class'] = create_balanced_bins(df['mmi'])

    categorical_features = ['alert', 'tsunami', 'net', 'magType', 'continent', 'country']
    label_encoders = {}
    for feature in categorical_features:
        if feature not in label_encoders:
            label_encoders[feature] = LabelEncoder()
        df[feature + '_encoded'] = label_encoders[feature].fit_transform(df[feature])

    numerical_features = ['magnitude', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']
    for feature in numerical_features:
        df[feature] = df[feature].fillna(df[feature].median())

    df['magnitude_squared'] = df['magnitude'] ** 2
    df['depth_magnitude_ratio'] = df['depth'] / df['magnitude']
    df['location_impact'] = df['depth'] * df['magnitude']

    numerical_features += ['magnitude_squared', 'depth_magnitude_ratio', 'location_impact']
    
    scaler = StandardScaler()
    scaler.fit(df[numerical_features])
    scaled_features = scaler.transform(df[numerical_features])
    X_numerical = scaled_features
    X_categorical = np.column_stack([df[f + '_encoded'] for f in categorical_features])

    le_cdi = LabelEncoder()
    le_mmi = LabelEncoder()

    y_cdi = le_cdi.fit_transform(df['cdi_class'])
    y_mmi = le_mmi.fit_transform(df['mmi_class'])

    return X_numerical, X_categorical, y_cdi, y_mmi, le_cdi.classes_, le_mmi.classes_

# Function to build the model
def build_model(num_numerical_features, num_categorical_features, num_cdi_classes, num_mmi_classes):
    numerical_input = Input(shape=(num_numerical_features,), name='numerical_input')
    x1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(numerical_input)
    x1 = residual_block(x1, 256)
    x1 = residual_block(x1, 256)

    categorical_input = Input(shape=(num_categorical_features,), name='categorical_input')
    x2 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(categorical_input)
    x2 = LayerNormalization()(x2)
    x2 = Dropout(0.3)(x2)

    combined = Concatenate()([x1, x2])
    x = residual_block(combined, 512)
    x = residual_block(x, 512)
    x = residual_block(x, 256)

    cdi_branch = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    cdi_branch = LayerNormalization()(cdi_branch)
    cdi_branch = Dropout(0.3)(cdi_branch)
    cdi_output = Dense(num_cdi_classes, activation='softmax', name='cdi_output')(cdi_branch)

    mmi_branch = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    mmi_branch = LayerNormalization()(mmi_branch)
    mmi_branch = Dropout(0.3)(mmi_branch)
    mmi_output = Dense(num_mmi_classes, activation='softmax', name='mmi_output')(mmi_branch)

    model = Model(
        inputs=[numerical_input, categorical_input],
        outputs=[cdi_output, mmi_output]
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.001
        ),
        loss={
            'cdi_output': 'sparse_categorical_crossentropy',
            'mmi_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'cdi_output': 1.0,
            'mmi_output': 1.0
        },
        metrics={
            'cdi_output': ['accuracy'],
            'mmi_output': ['accuracy']
        }
    )
    
    return model

# Residual block function
def residual_block(x, units):
    shortcut = x
    x = LayerNormalization()(x)
    x = Dense(units, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(units, kernel_regularizer=l2(0.01))(x)
    if shortcut.shape[-1] == units:
        x = Add()([x, shortcut])
    return x

# Function to evaluate the model
def evaluate_model(model, X_num, X_cat, y_cdi, y_mmi):
    y_pred_cdi, y_pred_mmi = model.predict([X_num, X_cat])
    y_pred_cdi = np.argmax(y_pred_cdi, axis=1)
    y_pred_mmi = np.argmax(y_pred_mmi, axis=1)

    st.subheader("Confusion Matrix for CDI")
    cm_cdi = confusion_matrix(y_cdi, y_pred_cdi)
    cm_cdi_df = pd.DataFrame(cm_cdi, columns=[f"Class {i}" for i in range(len(np.unique(y_cdi)))],
                              index=[f"Class {i}" for i in range(len(np.unique(y_cdi)))])
    st.write(cm_cdi_df)

    fig_cdi, ax_cdi = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_cdi, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(len(np.unique(y_cdi)))],
                yticklabels=[f"Class {i}" for i in range(len(np.unique(y_cdi)))])
    plt.title("Confusion Matrix for CDI")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig_cdi)

    st.subheader("Confusion Matrix for MMI")
    cm_mmi = confusion_matrix(y_mmi, y_pred_mmi)
    cm_mmi_df = pd.DataFrame(cm_mmi, columns=[f"Class {i}" for i in range(len(np.unique(y_mmi)))],
                              index=[f"Class {i}" for i in range(len(np.unique(y_mmi)))])
    st.write(cm_mmi_df)

    fig_mmi, ax_mmi = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_mmi, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(len(np.unique(y_mmi)))],
                yticklabels=[f"Class {i}" for i in range(len(np.unique(y_mmi)))])
    plt.title("Confusion Matrix for MMI")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig_mmi)

    st.subheader("Classification Report for CDI")
    report_cdi = classification_report(y_cdi, y_pred_cdi, output_dict=True)
    report_cdi_df = pd.DataFrame(report_cdi).transpose()
    st.write(report_cdi_df)

    st.subheader("Classification Report for MMI")
    report_mmi = classification_report(y_mmi, y_pred_mmi, output_dict=True)
    report_mmi_df = pd.DataFrame(report_mmi).transpose()
    st.write(report_mmi_df)

# Streamlit app setup
def render():
    st.title("Earthquake Prediction Model")
    st.markdown("""
            - This section allows you to train and evaluate a machine learning model that predicts the CDI (Community Internet Intensity Map) and MMI (Modified Mercalli Intensity) of an earthquake based on various features such as magnitude, depth, and geographical data. 
            - The model uses a combination of numerical and categorical data to make these predictions.

            - Click the "Show Results" button to load the dataset, train the model, and view the evaluation results, including confusion matrices, classification reports, and training history. 
            - This will help you understand how well the model performs in predicting earthquake intensity.
            """)

    if st.button("Show Results"):
        # Load data from the 'data' folder
        data_path = "data/finaldata.csv"
        try:
            df = pd.read_csv(data_path)
            st.write("Data Preview", df.head())

            # Initialize and preprocess data
            X_num, X_cat, y_cdi, y_mmi, cdi_classes, mmi_classes = preprocess_data(df)

            # Build and train the model
            model = build_model(
                num_numerical_features=X_num.shape[1],
                num_categorical_features=X_cat.shape[1],
                num_cdi_classes=len(cdi_classes),
                num_mmi_classes=len(mmi_classes)
            )

            # Train the model and store the history
            history = model.fit([X_num, X_cat], [y_cdi, y_mmi], epochs=100, batch_size=32, validation_split=0.2)

            st.success("Model trained successfully!")

            # Evaluate the model
            evaluate_model(model, X_num, X_cat, y_cdi, y_mmi)

            # Plot training history (Optional)
            st.subheader("Training Loss and Accuracy")
            fig_loss_accuracy = plt.figure(figsize=(12, 6))

            # Loss Plot
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            # Accuracy Plot
            plt.subplot(1, 2, 2)
            plt.plot(history.history['cdi_output_accuracy'], label='train accuracy (CDI)')
            plt.plot(history.history['val_cdi_output_accuracy'], label='val accuracy (CDI)')
            plt.plot(history.history['mmi_output_accuracy'], label='train accuracy (MMI)')
            plt.plot(history.history['val_mmi_output_accuracy'], label='val accuracy (MMI)')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            st.pyplot(fig_loss_accuracy)

        except FileNotFoundError:
            st.error(f"Data file not found at {data_path}. Please ensure the file is placed in the 'data' folder.")
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# ==============================
# 1. PREPROCESSING
# ==============================
def preprocess_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img / 255.0
    return img


# ==============================
# 2. CNN MODEL
# ==============================
def build_cnn():
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# ==============================
# 3. TRAIN CNN
# ==============================
def train_cnn(train_dir, test_dir):
    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )

    model = build_cnn()

    history = model.fit(train_data, epochs=10, validation_data=test_data)

    model.save("cnn_model.h5")

    return model, history


# ==============================
# 4. ML MODELS (COMPARISON)
# ==============================
def train_ml_models(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))


# ==============================
# 5. PREDICTION FUNCTION
# ==============================
def predict_image(model, img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    if pred[0][0] > 0.5:
        print("Tumor Detected")
    else:
        print("No Tumor")


# ==============================
# 6. YOLO (PLACEHOLDER)
# ==============================
def yolo_detection():
    print("YOLO detection module placeholder")
    print("Use Ultralytics YOLO separately with custom dataset")


# ==============================
# 7. U-NET (PLACEHOLDER)
# ==============================
def unet_segmentation():
    print("U-Net segmentation module placeholder")
    print("Implement separately for segmentation task")


# ==============================
# 8. MAIN FUNCTION
# ==============================
if __name__ == "__main__":

    print("Brain Tumor Detection System Started")

    # Set dataset paths
    train_dir = "dataset/train"
    test_dir = "dataset/test"

    # Train CNN
    model, history = train_cnn(train_dir, test_dir)

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Model Accuracy")
    plt.show()

    # Test prediction
    test_image = "test.jpg"  # replace with your image
    predict_image(model, test_image)

    print("Done.")

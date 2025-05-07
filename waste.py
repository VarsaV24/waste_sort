import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === STEP 1: Load & Prepare the Data ===

img_height, img_width = 150, 150
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

dataset_path = "/Users/Rithika/Desktop/garbage_classification"


train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === STEP 2: Build the CNN Model ===

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === STEP 3: Train the Model ===

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# === STEP 4: Evaluate Model Performance ===

val_preds = model.predict(val_data)
y_pred = np.argmax(val_preds, axis=1)
y_true = val_data.classes

# Classification report
report = classification_report(y_true, y_pred, target_names=val_data.class_indices.keys(), output_dict=True)

# Extract metrics for bar chart
metrics = {
    "Accuracy": history.history['val_accuracy'][-1],
    "Precision": precision_score(y_true, y_pred, average='macro'),
    "Recall": recall_score(y_true, y_pred, average='macro'),
    "F1 Score": f1_score(y_true, y_pred, average='macro')
}

# === STEP 5: Plot Bar Chart ===

plt.figure(figsize=(8,5))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
# train_disease.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "PlantVillage")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.h5")
LOG_PATH = os.path.join(BASE_DIR, "logs", "disease_model_log.txt")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(224,224), batch_size=32, class_mode="categorical", subset="training"
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=(224,224), batch_size=32, class_mode="categorical", subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=10)

loss, acc = model.evaluate(val_gen)

model.save(MODEL_PATH)

with open(LOG_PATH, "w") as f:
    f.write(f"Validation Accuracy: {acc:.2f}\n")
    f.write(f"Validation Loss: {loss:.2f}\n")

print(f"✅ Disease model trained with Accuracy: {acc:.2f}")
print(f"📂 Logs saved at {LOG_PATH}")



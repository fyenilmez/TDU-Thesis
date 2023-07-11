
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import Dense,Dropout,Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import load_model

random.seed(42)
tf.random.set_seed(42)

validation_dir = 'faces_label_validation/'
training_dir = 'faces_labeled_same_as_fer2013_training_set_new/'
test_dir = 'deneme-test-face'
# Set the input shape for the VGG19 model
input_shape = (224, 224, 3)

# Set up the data generator with augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)


# Set up the data generator without augmentation for validation data
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
num_classes = 6

# Set up the training and validation data generators
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Set the number of training and validation samples
num_train_samples = len(train_generator.filenames)
num_val_samples = len(val_generator.filenames)

# Set the number of training and validation steps
train_steps = num_train_samples // batch_size
val_steps = num_val_samples // batch_size

# Create the VGG19 model with pre-trained weights
base_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, pooling='avg',input_shape=input_shape)
base_input = base_model.layers[0].input
base_output = base_model.layers[-3].output
model = Model(base_input, base_output)
model.summary()


# Add your own classification layers on top of the pre-trained model
x = GlobalAveragePooling2D()(base_output)
x = Dense(1024, activation='relu')(x)
out = Dense(6, activation='softmax', name='classifier')(x)#7 classes in FER2013

model = Model(base_input, out)


model.summary()

# Compile the model
sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

filepath="Faces_ResNet152V2_weights_improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)

callbacks_list = [checkpoint, reduce_lr, early_stop]

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=200,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[callbacks_list]
)

# Load the weights of the best validation accuracy model
model.load_weights('Faces_EfficientNetV2B1_weights.hdf5') 

# Compile the model with the same optimizer, loss function, and metrics as during training
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# Define the data generator for testing
test_datagen_faces = ImageDataGenerator(rescale=1./255)


# Set the input image size
input_size = (224, 224)

# Set the batch size for testing
batch_size = 32

# Create the test data generator
test_generator_faces = test_datagen_faces.flow_from_directory(
    test_data_dir_faces,
    target_size=input_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Get the total number of test samples
num_test_samples = test_generator_faces.samples

# Predict the labels for the test data
predictions = model.predict_generator(test_generator_faces, steps=np.ceil(num_test_samples / batch_size))

# Get the predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Get the true labels
true_labels = test_generator_faces.classes

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Calculate accuracy for each class
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Display accuracy for each class
for class_label, accuracy in enumerate(class_accuracy):
    print(f"Class {class_label}: Accuracy = {accuracy:.2%}")

overall_accuracy = cm.diagonal().sum() / cm.sum()
print(f"Overall Accuracy: {overall_accuracy:.2%}")


# Evaluate the model on the test set
train_loss, train_accuracy = model.evaluate(train_generator)
print('Train loss:', train_loss)
print('Train accuracy:', train_accuracy)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_accuracy)


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')


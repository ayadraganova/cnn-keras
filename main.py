import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

'''
Preparing Test & Training data
'''
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,
                                                                                             target_size=(224, 224),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10)

valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                             target_size=(224, 224),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10)

test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                             target_size=(224, 224),
                                                                                             classes=['cat', 'dog'],
                                                                                             batch_size=10)

imgs, labels = next(train_batches)


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# plot_images(imgs)
# print(labels)

'''
Build and train the CNN
'''
model = Sequential(
    [Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
     MaxPool2D(pool_size=(2, 2), strides=2),
     Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
     MaxPool2D(pool_size=(2, 2), strides=2),
     Flatten(),
     Dense(units=2, activation='softmax')])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

'''
Predict
'''
test_imgs, test_labels = next(test_batches)
plot_images(test_imgs)
print(test_labels)
test_batches.classes

predictions = model.predict(x=test_batches, verbose=0)
np.round(predictions)
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


test_batches.class_indices

cm_plot_labels = ['cat', 'dog']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels)

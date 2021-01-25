# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Below is code
with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad.Create a convolutional neural network that trains to 100 % accuracy on these images, which cancels training upon hitting training accuracy of > .999

Hint - - it
will
work
best
with 3 convolutional layers.
"""
import tensorflow as tf
import os
import zipfile

DESIRED_ACCURACY = 0.999

# !wget - -no - check - certificate \
#     "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#     - O
# "/tmp/happy-or-sad.zip"

# zip_ref = zipfile.ZipFile(r"C:\Users\michael.guerrero\PycharmProjects\cert_prac\tensorflow_cert_prac\tmp\happy-or-sad.zip", 'r')
# zip_ref.extractall(r"C:\Users\michael.guerrero\PycharmProjects\cert_prac\tensorflow_cert_prac\tmp\h-or-s")
# zip_ref.close()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("accuracy = {}".format(logs.get('accuracy')))
        print("acc  = {}".format(logs.get('acc')))
        if (logs.get('accuracy') > DESIRED_ACCURACY):
            print("\nReached {}% accuracy so cancelling training!".format(DESIRED_ACCURACY*100))
            self.model.stop_training = True#
callbacks=myCallback()
# # This Code Block should Define and Compile the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
#
#
# from tensorflow.keras.optimizers import RMSprop

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
#
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# This code block should create an instance of an ImageDataGenerator called train_datagen
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\michael.guerrero\PycharmProjects\cert_prac\tensorflow_cert_prac\tmp\h-or-s",
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

    # Expected output: 'Found 80 images belonging to 2 classes'
# This code block should call model.fit and train for
# a number of epochs.
history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[callbacks])

# Expected output: "Reached 99.9% accuracy so cancelling training!""

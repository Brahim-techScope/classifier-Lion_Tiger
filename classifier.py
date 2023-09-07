import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

# You will need these two directories
# Directory with our training lion pictures
train_lion_dir = os.path.join('./lion_tiger/train/lions')

# Directory with our training tiger pictures
train_tiger_dir = os.path.join('./lion_tiger/train/tigers')

# Directory with our testing lion pictures
test_lion_dir = os.path.join('./lion_tiger/test/lions')

# Directory with our testing tiger pictures
test_tiger_dir = os.path.join('./lion_tiger/test/tigers')

# Number of training data
print('total training lion images:', len(os.listdir(train_lion_dir)))
print('total training tiger images:', len(os.listdir(train_tiger_dir)))

# Number of testing data
print('total training lion images:', len(os.listdir(test_lion_dir)))
print('total training tiger images:', len(os.listdir(test_tiger_dir))) # Testing data represents 20% of all the data

# Displaying an image from data
image = os.listdir(train_lion_dir)[0]
def display_image(image):
    img_path = os.path.join(train_lion_dir, image)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()

# Prepare the data
# Configure an image data generator for training. All images will be rescaled by 1./255. I applied some data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, rotation_range=40,horizontal_flip=True,shear_range=0.2)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './lion_tiger/train',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=24,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# Flow testing images in batches of 24 using test_datagen generator
test_generator = test_datagen.flow_from_directory(
        './lion_tiger/test',
        target_size=(300, 300),
        batch_size=24,
        class_mode='binary')


# look if there is any image of shape (300, 300, 4)
def check_for_4():
    for i in range(len(train_generator)):
        batch_images, batch_labels = train_generator.next()
        
        # Iterate through the images in the batch
        for j in range(len(batch_images)):
            image = batch_images[j]
            
            # Check if the image has the shape (300, 300, 4)
            if image.shape == (300, 300, 4):
                print("Image with 4 color bytes filename:", train_generator.filenames[i * train_generator.batch_size + j])

# The structure of the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(300, 300, 3)),
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 128 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for class ('lions') and 1 for the other ('tigers')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Define a custom callback class by subclassing tf.keras.callbacks.Callback
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.95):
        super(CustomEarlyStopping, self).__init__() # Call the constructor of the parent class when an object is cretaed
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Logs containes informations like accuracy and val_accuracy
        if logs is None:
            logs = {}
        
        val_accuracy = logs.get('val_accuracy')
        accuracy = logs.get('accuracy')
        
        if val_accuracy is not None and accuracy is not None:
            if val_accuracy >= self.threshold and accuracy >= self.threshold:
                print(f"Reached both val_accuracy ({val_accuracy}) and accuracy ({accuracy}) >= {self.threshold}. Stopping training.")
                self.model.stop_training = True

# Create an instance of the custom callback
custom_early_stopping_callback = CustomEarlyStopping(threshold=0.95)

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

# Train the model using the training generator
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20, # 20 times passing through the entire dataset
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[custom_early_stopping_callback],
    verbose=1 # See detailed progress and performance information for each epoch during training
)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('optimized_model.tflite', 'wb') as f:
    f.write(tflite_model)
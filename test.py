#import tensorflow as tf
import os 
import pickle

import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage import transform 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow
def print_img(image28):
    # Get the unique labels 
    unique_labels = set(labels)

    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1


      # For each unique label,
    for label in unique_labels:
        # You pick the first image for each label
        image = images28[labels.index(label)]
        # Define 64 subplots 
        plt.subplot(8, 8, i)
        # Don't include axes
        plt.axis('off')
        # Add a title to each subplot 
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        # Add 1 to the counter
        i += 1
        # And you plot this first image 
        plt.imshow(image,  cmap="gray")


    # Show the plot
    plt.savefig('signs_gray.jpg')


'''
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/lfu/Downloads"
train_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
test_data_directory = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

images, labels = load_data(train_data_directory)
# Print the `images` dimensions
images = np.array(images)
images28 = [transform.resize(image, (28, 28)) for image in images]




# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

pickle.dump( (images28, labels), open( "save.p", "wb" ) )
'''
num = int(input("epoch nums:"))
(images28, labels) = pickle.load(  open( "save.p", "rb" ) )
#print_img(images28)

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tensorflow.contrib.layers.flatten(x)

# Fully connected layer 
logits = tensorflow.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


tf.set_random_seed(17)
sess = tf.Session()
sess.run(tf.global_variables_initializer())



'''
for i in range(300):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')
saver = tf.train.Saver()
save_path = saver.save(sess, "model.ckpt")
'''
for i in range(int(num)):
        _, accuracy_val, loss_val = sess.run([train_op, accuracy, loss], feed_dict={x: images28, y: labels})
        if i % 100 == 0:
            print("                                     Loss: ", loss_val)


'''
import random

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.savefig("result.jpg")
'''

test_images28,test_labels =  pickle.load(  open( "save.p", "rb" ) )

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()
print("Done")
'''
# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images that you defined 
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)

plt.show()
'''

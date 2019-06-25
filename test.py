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

num = int(input("epoch nums:"))
(data, labels) = pickle.load(  open( "flatten", "rb" ) )

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None,200])
y = tf.placeholder(dtype = tf.int32, shape = [None])


# Fully connected layer 
logits = tensorflow.contrib.layers.fully_connected(data, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
#train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
tf_learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
optimalizer = tf.train.AdamOptimizer(learning_rate=0.001)#MomentumOptimizer(tf_learning_rate,momentum=0.9)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

grads_and_vars = optimalizer.compute_gradients(loss)
train_op = optimalizer.minimize(loss)


tf.set_random_seed(17)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','first')):
    os.mkdir(os.path.join('summaries','first'))

summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), sess.graph)




# Name scope allows you to group various summaries together
# Summaries having the same name_scope will be displayed on the same row
with tf.name_scope('performance'):
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

    # Whenever you need to record the loss, feed the mean test accuracy to this placeholder
    tf_accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    # Create a scalar summary object for the accuracy so it can be displayed
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

# Gradient norm summary
for g,v in grads_and_vars:
    if 'hidden5' in v.name and 'weights' in v.name:
        with tf.name_scope('gradients'):
            tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
            tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
            break
# Merge all summaries together
performance_summaries = tf.summary.merge([tf_loss_summary,tf_accuracy_summary])

summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), sess.graph)


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
        if i == 0:
            _, accuracy_val, loss_val = sess.run([train_op, accuracy, loss], feed_dict={x: images28, y: labels, tf_learning_rate: 0.0001})
          #  summ_writer.add_summary(loss_val, i)
            

        _, accuracy_val, loss_val = sess.run([train_op, accuracy, loss], feed_dict={x: images28, y: labels, tf_learning_rate: 0.0001})
        if i % 100 == 0:
            print("                                     Loss: ", loss_val)





test_data,test_labels =  data[25000:], labels[25000:]

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_data})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)


# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()
print("Done")


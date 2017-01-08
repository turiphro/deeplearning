# Hello World app for TensorFlow

# Notes:
# - TensorFlow is written in C++ with good Python (and other) bindings.
#   It runs in a separate thread (Session).
# - TensorFlow is fully symbolic: everything is executed at once.
#   This makes it scalable on multiple CPUs/GPUs, and allows for some
#   math optimisations. This also means derivatives can be calculated
#   automatically (handy for SGD).

import tensorflow as tf

# define the graph
M1 = tf.constant([[3., 3.]])
M2 = tf.constant([[2.], [2.]])
M3 = tf.matmul(M1, M2) # symbolic: no calculation yet, all happens at once outside of Python (in GPU, on network, etc)

# start a session to compute the graph
with tf.Session() as sess: # runs on GPU first
    #with tf.device("/gpu:1"): # explicitly choose if you have multiple GPUs
    #with tf.device("grpc://host:2222"): # explicitly choose host with running TensorFlow server
    result = sess.run(M3) # runs subsection of total graph
    print(result) # [[12.]]

state = tf.Variable(0, name='counter')  # maintains state along Session
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)    # again symbolic
init_op = tf.initialize_all_variables() # makes operator; does not run anything yet

with tf.Session() as sess:      # start process that runs TensorFlow
    sess.run(init_op)           # run init vars part of graph
    print(sess.run(state))      # run creates state
    for _ in range(3):
        print(sess.run(update)) # run (part of) graph for updates

# to allow setting inputs for each flow, use placeholders:
#x = tf.placeholder(tf.float32, [None, 784]) # placeholder = input
#sess.run(x, feed_dict={x: mnist.test.images})

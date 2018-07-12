import tensorflow as tf
import numpy as np
x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

# with tf.control_dependencies([x_plus_1]):
#     y =tf.identity(x)
y=x_plus_1
init = tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)
for i in range(5):
    print(sess.run(y))
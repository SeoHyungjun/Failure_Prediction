import tensorflow as tf
import numpy as np

x = np.array([[0,1,2], [3,4,5], [6,7,8], [9,10,11]])

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(4))


print("x = \n{}".format(x))

print("shuffle_indices = {}".format(shuffle_indices))

x_shuffled = x[shuffle_indices]
print("x_shuffled = \n{}".format(x_shuffled))


x = tf.constant(np.array([[0.0,1.0,2.0], [3.0,4.0,5.0]]))
x_soft = tf.nn.softmax(x, -1)
sess = tf.Session()
print(sess.run(x_soft))


a = [[1],[2]]
a_array = np.array(a)
a.append([3])
print(a)
print(a_array.shape)


num_filters_total = 6
num_classes = 4
l2_loss = tf.constant(0.0)
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape = [num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    l2 = tf.nn.l2_loss(x)
#    self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    global_step = tf.Variable(0, name="global_step", trainable=False)

init_op = tf.initialize_all_variables()
sess.run(init_op)
print(sess.run(W))
print(sess.run(l2))

print(sess.run(global_step))

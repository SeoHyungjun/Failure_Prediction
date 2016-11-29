import tensorflow as tf

i = 1
dimen = 0

arr1 = tf.constant([1,2,3,4])
arr2 = tf.constant([[1,2,3,4], [5,6,7,8], [9, 10, 11, 12]])
arr3 = tf.constant([[[1,2,3,4],
		  [5,6,7,8],
		  [9, 10, 11, 12]], 
	         [[13, 14, 15, 16], 
		  [17, 18, 19, 20], 
		  [21, 22, 23, 24]]])

arr = [None, arr1, arr2, arr3]

#arr = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5]])
# argmax(arr, 0) = [1 1 1 1],  argmax(arr, 1) = [3 3]
#arr = tf.constant([2, 3, 4, 5])   # argmax(arr, 0) = 3


while True:
    print("input index num(1~3). if 'i = 0', exit.")
    i = int(input())
    if i == 0:
        break
    print("input dimension(0~(i-1))")
    dimen = int(input())
    argmax = tf.argmax(arr[i], dimen)
    with tf.Session() as sess:
        print("----------------------------------")
        print("arr[%d]: \n%s" % (i, sess.run(arr[i])))
        print("\nargmax(dimen : %d) = \n%s" % (dimen, sess.run(argmax)))
        print("----------------------------------\n\n")



""" tf.eval()
sess = tf.Session()
with sess.as_default():
    print(argmax.eval())
"""

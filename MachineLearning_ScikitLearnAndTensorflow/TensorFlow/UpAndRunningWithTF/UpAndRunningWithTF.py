#Creating and running a graph
import tensorflow as tf

tf.reset_default_graph()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2


sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()


init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()


sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)

#Managing graphs

tf.reset_default_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph

x2.graph is tf.get_default_graph()
import numpy as np
import tensorflow as tf


class Classifier:
	def __init__(self, classifier, save_path='', dropout=False):
		self.x = tf.placeholder('float', [None, 3])
		self.y = tf.placeholder('float', [None, 10])
		self.model = self.build_model(dropout)
		self.classifier = classifier
		self.save_path = save_path

	def build_model(self, dropout):
		weights = {
			'h1': tf.Variable(tf.random_normal([134, 256])),
			'h2': tf.Variable(tf.random_normal([256, 512])),
			'h3': tf.Variable(tf.random_normal([512, 256])),
			'out': tf.Variable(tf.random_normal([256, 10]))
		}
		biases = {
			'b1': tf.Variable(tf.random_normal([256])),
			'b2': tf.Variable(tf.random_normal([512])),
			'b3': tf.Variable(tf.random_normal([256])),
			'out': tf.Variable(tf.random_normal([10]))
		}

		layer1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
		if dropout:
			layer1 = tf.nn.dropout(layer1, 0.5)
		layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
		if dropout:
			layer2 = tf.nn.dropout(layer2, 0.5)
		layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
		return tf.matmul(layer3, weights['out']) + biases['out']

	def train(self, training_data, epochs):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
		init, saver = tf.initialize_all_variables(), tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(epochs):
				batch_x, batch_y = [m[0] for m in training_data], [n[1] for n in training_data]
				_, avg_cost = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
				if epoch % 500 == 0:
					print 'Classifier %d' % self.classifier, ' Epoch', '%04d' % epoch, 'cost = %04f' % avg_cost
			saver.save(sess, self.save_path) if self.save_path != '' else ''

	def classify(self, input_data):
		init, saver = tf.initialize_all_variables(), tf.train.Saver()
		with tf.Session() as sess:
			sess.run(init)
			saver.restore(sess, self.save_path)
			classification = np.asarray(sess.run(self.model, feed_dict={self.x: input_data}))
			return np.unravel_index(classification.argmax(), classification.shape)
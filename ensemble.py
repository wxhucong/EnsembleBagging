import random
import numpy as np
from classifier import Classifier


class Ensemble:
	def __init__(self, save_dir, num_classifiers):
		self.classifiers = []
		for i in range(num_classifiers):
			classifier = Classifier(i, save_dir + '/model' + str(i) + '.cptk')
			self.classifiers.append(classifier)

	def train(self, training_data, epochs):
		for classifier in self.classifiers:
			data = random.sample(training_data, len(training_data) / 2)
			classifier.train(data, epochs)

	def classify(self, features):
		results = np.zeros(10)
		for classifier in self.classifiers:
			results[classifier.classify(features)] += results[classifier.classify(features)]
		return np.unravel_index(results.argmax(), results.shape)
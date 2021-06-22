"""
File: titanic_level1.py
Name: Kevin Chen
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
import util
from itertools import combinations_with_replacement
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: dict[str: list], key is the column name, value is its data
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	############################
	with open(filename, 'r') as f:
		first = True
		for line in f:
			lst = line.strip().split(',')
			if mode == 'Train':
				if first:
					# ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
					for i in (1, 2, 4, 5, 6, 7, 9, 11):
						data[lst[i]] = []
					first = False
				else:
					lst = [lst[i] for i in (1, 2, 5, 6, 7, 8, 10, 12)]
					if '' not in lst:
						for i in range(len(lst)):
							if i == 0:
								data['Survived'].append(int(lst[i]))
							if i == 1:
								data['Pclass'].append(int(lst[i]))
							elif i == 2:
								if lst[i] == 'female':
									data['Sex'].append(0)
								else:
									data['Sex'].append(1)
							elif i == 3:
								data['Age'].append(float(lst[i]))
							elif i == 4:
								data['SibSp'].append(int(lst[i]))
							elif i == 5:
								data['Parch'].append(int(lst[i]))
							elif i == 6:
								data['Fare'].append(float(lst[i]))
							elif i == 7:
								if lst[i] == 'S':
									data['Embarked'].append(0)
								elif lst[i] == 'C':
									data['Embarked'].append(1)
								else:
									data['Embarked'].append(2)

			elif mode == 'Test':
				if first:
					# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
					for i in (1, 3, 4, 5, 6, 8, 10):
						data[lst[i]] = []
					first = False
				else:
					lst = [lst[i] for i in (1, 4, 5, 6, 7, 9, 11)]
					for i in range(len(lst)):
						if i == 0:
							data['Pclass'].append(int(lst[i]))
						elif i == 1:
							if lst[i] == 'female':
								data['Sex'].append(0)
							else:
								data['Sex'].append(1)
						elif i == 2:
							if lst[i] == '':
								data['Age'].append(round(sum(training_data['Age'])/len(training_data['Age']), 3))
							else:
								data['Age'].append(float(lst[i]))
						elif i == 3:
							data['SibSp'].append(int(lst[i]))
						elif i == 4:
							data['Parch'].append(int(lst[i]))
						elif i == 5:
							if lst[i] == '':
								data['Fare'].append(round(sum(training_data['Fare'])/len(training_data['Age']), 3))
							else:
								data['Fare'].append(float(lst[i]))
						elif i == 6:
							if lst[i] == 'S':
								data['Embarked'].append(0)
							elif lst[i] == 'C':
								data['Embarked'].append(1)
							else:
								data['Embarked'].append(2)
	############################
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	n = 0
	for key in sorted(set(data[feature])):
		data[f'{feature}_{n}'] = []
		for i in range(len(data[feature])):
			if data[feature][i] == key:
				data[f'{feature}_{n}'].append(1)
			else:
				data[f'{feature}_{n}'].append(0)
		n += 1
	data.pop(feature)
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	for key in data:
		x_min = min(data[key])
		x_max = max(data[key])
		new_data_list = []
		for x in data[key]:
			new_data_list.append((x - x_min) / (x_max - x_min))
		data[key] = new_data_list
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0

	# Step 2 : Feature Extract
	if degree == 1:
		data = inputs.copy()
	elif degree == 2:
		data = inputs.copy()
		for x1, x2 in list(combinations_with_replacement(keys, 2)):
			data[x1 + x2] = []
			for i in range(len(labels)):
				data[x1 + x2].append(inputs[x1][i]*inputs[x2][i])

	# Step 3 : Start training & Update weights
	for i in range(num_epochs):
		for j in range(len(labels)):
			f = {}
			for key in list(data.keys()):
				f[key] = data[key][j]
			util.increment(weights, - alpha * (sigmoid(util.dotProduct(weights, f)) - labels[j]), f)
	return weights


def sigmoid(k):
	return 1/(1+math.exp(-k))

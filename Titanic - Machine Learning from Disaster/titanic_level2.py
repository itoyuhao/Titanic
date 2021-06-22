"""
File: titanic_level2.py
Name: Kevin Chen
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None

	if mode == 'Train':
		data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
		data = data.dropna()
		labels = data.pop('Survived')

	elif mode == 'Test':
		data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
		# fill the missing data with mean value of training data
		data['Age'] = data['Age'].fillna(round(training_data['Age'].mean(), 3))
		data['Fare'] = data['Fare'].fillna(round(training_data['Fare'].mean(), 3))

	# Changing 'male' to 1, 'female' to 0 with loc
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	n = 0
	for key in sorted(set(data[feature])):
		data[f'{feature}_{n}'] = 0
		data.loc[data[feature] == key, f'{feature}_{n}'] = 1
		n += 1
	data.pop(feature)
	# print(data.head(8))
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83707865
	TODO: real accuracy on degree3 -> 0.87640449
	"""
	# Data Cleaning
	train_data, true_label = data_preprocess(TRAIN_FILE, mode='Train')
	# print(train_data.head(6))
	test_data = data_preprocess(TEST_FILE, mode='Test', training_data=train_data)
	# print(test_data.head(6))

	# One-hot Encoding & standardization
	for data in train_data, test_data:
		one_hot_encoding(data, 'Sex')
		one_hot_encoding(data, 'Pclass')
		one_hot_encoding(data, 'Embarked')

	# Standardization
	standardizer = preprocessing.StandardScaler(copy=True)
	train_data = standardizer.fit_transform(train_data)
	test_data = standardizer.transform(test_data)

	# Load Logistic Regression Model
	h = linear_model.LogisticRegression(max_iter=10000)

	# ---------------------------------------------------------------------#

	# Degree 1 polynomial - Training
	classifier = h.fit(train_data.copy(), true_label)
	acc = classifier.score(train_data.copy(), true_label)
	print('Training Acc(Degree 1):', round(acc, 8))

	# Degree 1 polynomial - Testing
	pred = classifier.predict(test_data.copy())
	# print('Prediction Acc:\n', pred)

	# ---------------------------------------------------------------------#

	# Degree 2 polynomial - Training
	poly_feature_extractor = PolynomialFeatures(degree=2)
	x_poly = poly_feature_extractor.fit_transform(train_data.copy())
	classifier_poly = h.fit(x_poly, true_label)
	acc = classifier_poly.score(x_poly, true_label)
	print('Training Acc(Degree 2):', round(acc, 8))

	# Degree 2 polynomial - Testing
	test_data_poly = poly_feature_extractor.transform(test_data)
	pred = classifier_poly.predict(test_data_poly)
	# print('Prediction Acc(Degree 2):\n', pred)

	# ---------------------------------------------------------------------#

	# Degree 3 polynomial - Training
	poly_feature_extractor = PolynomialFeatures(degree=3)
	x_poly = poly_feature_extractor.fit_transform(train_data.copy())
	classifier_poly = h.fit(x_poly, true_label)
	acc = classifier_poly.score(x_poly, true_label)
	print('Training Acc(Degree 3):', round(acc, 8))

	# Degree 3 polynomial - Testing
	test_data_poly = poly_feature_extractor.transform(test_data)
	pred = classifier_poly.predict(test_data_poly)
	# print('Prediction Acc(Degree 3):\n', pred)



if __name__ == '__main__':
	main()

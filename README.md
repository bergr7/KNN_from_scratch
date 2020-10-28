# K-Nearest Neighbors from scratch

## Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Instructions
5. Results
6. Licensing, Authors, Acknowledgements
7. MIT License

## Installation

- Libraries included in Anaconda distribution of Python 3.8.

## Project Motivation

The aim of this project is to get familiar with KNN implementation from scratch.

## File Descriptions

- knn.py - This python script contains a class called Knn() that is a classifier that implements the K-Nearest Neighbors
vote.
- test.py - This python script contain several tests that the Knn class has to pass. Most of them compare Knn() with
results obtained with Sklearn KNeighborsClassifier.

## Instructions

Run the following command in the the terminal in your working directory to check that Knn passes all the tests:

    python test.py
    
In order to use Knn(), it can be imported from knn.py module as you would import KNeighborsClassifier from Sklearn:

    from knn import Knn
    
Then you can instantiate Knn() in a variable, specify arguments and use its methods. For example:

    # instantiate Knn
    clf = Knn(n_neighbors=4, metric='minkowski', p=2, weights='uniform')
    
    # fit Knn on training data
    clf.fit(X_train, y_train)
    
    # make predictions
    y_pred = clf.predict(X_test)
    
    # display a confusion matrix
    clf.display_results(y_test, y_preds)

## Results

This simple implementation of KNN from scratch allows users to get familiar with how Sklearn KNeighborsClassifier works
under the hood.

All the code is in a public repository at the link below:

https://github.com/bergr7/KNN_from_scratch

## Licensing, Authors and Acknowledgements

I would like to give credit to:

- M. Reza Zerehpoosh for his article "How to implement k-Nearest Neighbors (KNN) classifier from scratch in Python".
- Stackoverflow community for making code questions and answers publicly available
- Jason Brownlee from Machine Learning Mastery for the post "Develop k-Nearest Neighbors in Python From Scratch".
- Miguel Camacho from Big Data International Campus and UCAM for proposing this learning project.

## MIT License

Copyright 2020 Bernardo Garcia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

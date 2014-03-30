iDML
===

D's Machine Learning is a machine learning toolkit for python,focus on rightness but efficiency

all code is based on numpy and scipy

----------------------------------------------
Code Files
===

`./dml/NN` -the code of **Neural NetWorks**

`./dml/LR` -**Logistic Regression**,actualy It's **softmax**

`./dml/DT` -**Decision Tree** , CART algorithm

`./dml/ClUSTER` -some cluster algorithm,inculde **kmeans \ kmedoids \ spectralCluster \ Hierarchical Cluster**

`./dml/ADAB` -the **adaboost** algorithm

`./dml/KNN` -the **k-Nearest Neighbor** algorithm(kd-tree BBF implementing)

`./dml/NB`  -the **naive Bayesian** support both  continous and descrete features

`./dml/SVM` -the basic binary **Support Vector Machine**

`./dml/CNN` -the simple **Convolutional Neural Networks**

`./dml/CF` -some **Collaborative Filtering Algorithm** implement,include **item-based \ SVD \ RBM**

`./dml/tool` -include some basic tools for computing

`./test/` -include some **test code** for DML

----------------------------------------------
Class Format
===
all class can be used in this way:(LR for example)

but there is still some different Initialization parameters in different class,also the predict function

sorry for this but most class use `pred()` and NN use `nnpred()`,I may formalize them in  the future

    a = LRC(train_images,trian_labels,nor=False)
	a.train(200,True)
	pred = a.predict(test_images)
	
for the input  X and y  ,**X** must be a **N\*M matrix** and 
**y** is a **vector length M**

where  N is  the  `#feature` and  M is `#training_case`


for the cluster method,you can use `a.labels` or `a.result()` to get the final result

----------------------------------------------
Install
===
DML is based on `numpy`,`scipy`,`matplotlib`   .you should install them first

This packages uses setuptools, which is the default way of installing python modules. The install command is:(sudo is required in some system)

	python setup.py build
	python setup.py install

----------------------------------------------
Warning
===
* only python 2 is supported,sorry for the python 3 user.

* some method from numpy and  scipy will report warning because of their version

----------------------------------------------
License
===
[WTFPL](http://www.wtfpl.net/)

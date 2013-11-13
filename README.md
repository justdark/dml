DML
===

D's Machine Learning is a machine learning toolkit for python,focus on rightness but efficiency


all code is based on numpy and scipy

----------------------------------------------
Code Files
===

`./dml/NN` -the code of Neural NetWorks

`./dml/LR` -Logistic Regression,actualy It's softmax

`./dml/DT` -Decision Tree , CART algorithm

`./dml/ClUSTER` -some cluster algorithm,inculde kmeans \ kmedoids \ spectralCluster

`./dml/ADAB` -the adaboost algorithm

`./dml/KNN` -the k-Nearest Neighbor algorithm(kd-tree BBF implementing)

`./dml/tool` -include some basic tools for computing

`./test/` -include some test code for DML

----------------------------------------------
Class Format
===
all class can be used in this way:(LR for example)

but there is still some different Initialization parameters in different class,also the predict function

sorry for this but some class use `pred()` and NN use `nnpred()`

    a = LRC(train_images,trian_labels,nor=False)
	a.train(200,True)
	pred = a.predict(test_images)
	

for the cluster method,you can use `a.labels` or `a.result()` to get the final result

----------------------------------------------
Install
===
DML is based on `numpy`,`scipy`,you should install them first

This packages uses setuptools, which is the default way of installing python modules. The install command is:(sudo is required in some system)

	python setup.py build
	python setup.py install


----------------------------------------------
License
===
[WTFPL](http://www.wtfpl.net/)

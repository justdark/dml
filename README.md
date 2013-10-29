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

`./dml/ClUSTER` -some cluster algorithm,inculde kmeans\kmedoids\spectralCluster(todo)

`./test/` -include some test code for DML

----------------------------------------------
Class Format
===
all class can be used in this way:(LR for example)

but there is still some different Initialization parameters in different class

    a = LRC(train_images,trian_labels,nor=False)
	a.train(200,True)
	pred = a.predict(test_images)
	

----------------------------------------------
License
===
[WTFPL](http://www.wtfpl.net/)

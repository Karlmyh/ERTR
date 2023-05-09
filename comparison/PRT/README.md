#### IMPORTANT NOTE:
This is readme for probabilistic regression trees, for a readme on the scikit-learn
library, please refer to **README_SKLEARN.rst**

## Probabilistic Regression Trees
Probabilistic regression trees run on python as a part of the scikit-learn library, and to run the application, python 3 is needed.
Once python 3 is installed, run the following commands to install the required packaged:
```
pip install numpy
pip install pandas
pip install multiprocessing
pip install scipy
pip install statsmodels
pip install Cython
```

### Compiling scikit-learn
With the packages installed, scikit-learn needs to be compiled since it is build on cython, 
you can simply compile scikit-learn with the following commands inside the probabilistic-tree folder:
```
python3 setup.py clean
python3 setup.py install build_ext -i
```
### PR Application
Included in the code three files that can be used to test the algorithm, 
* **pr_tree_toy_example**: a simple toy example to test the algorithm
* **pr_tree_main**: contains the code to run a k-fold-cross-validation (train-test) on a certain dataset. Validation is not included in this script.
* **pr_tree_validator**: contains the code to run a k-fold-cross-validation (train-validation-test) on a certain dataset. Validation steps are included in this script.

#### Parameters:
```
-x X        x path
-s S        splitting method (topk1, topk3, topk5), default=topk3
-m M        criterion method (mseprob for uncertain_tree), mse for standard trees, default=mseprob
-l L        min leaf percentage [0, 1], default=0.1
-ts TS      test size in percentage [0, 1], default=0.2
-cvs CVS    Cross validation start index (for main and validator), default=0
-cve CVE    Cross validation end index (for main and validator), default=10
-sg SG		noise modifier for each cross validation separated by, 
			this can be used if the user knows the noise level of each fold. 
			If not given, the standard deviation is used for all folds (main only)
```

#### Dataset description
For the sake of simplicity, the structure of the given dataset needs to follow a simple csv structure,
where the rows are the observations (instances) and the columns are the features.
The target column (Y) needs to be the last column

#### Toy Example with Boston dataset
To run the toy example, run the following command for probabilistic tree with top V=3, V=5 and Std:
```
python3 pr_tree_toy_example.py -x dataset/Boston.csv -m mseprob -s topk3
RMSE: 3.878920069166821
```
```
python3 pr_tree_toy_example.py -x dataset/Boston.csv -m mseprob -s topk5
RMSE: 3.316435004000185
```
Standard tree:
```
python3 pr_tree_toy_example.py -x dataset/Boston.csv -m mse
RMSE: 5.083519584975742
```
### Reproduce  the Results
To reproduce the results indicated in the paper, you can simply run the validator script on any dataset wanted. Four datasets were included in the zip file that can be tested easily, for example, to reproduce the exact same RMSE for Boston (4.47±1.04):
```
python3 pr_tree_validator.py -x dataset/Boston.csv
```
The output will follow the structure "k (CV fold), noise_modifier, RMSE", e.g.:
```
0 1e-20 4.763
0 0.25 4.146
0 0.5 3.801
0 0.75 3.634
0 1 3.595
0 1.25 3.567
0 1.5 3.772
0 1.75 4.009
0 2 4.053
Best 0 1.25 5.460
...
Best 9 1 4.239

Avg: 4.467
Std: 1.041
```

As you can see from the previous example, we run the validator to choose the best sigma_u (noise) modifier on the validation set, then re-build the model with the best modifier on the testset, for instance in the previous example, the best modifier for fold 0 is 1.25, while for fold 9 it is 1.

If the user knows the level of noise and wants to re-run the experiments, the main script can be used with -sg option:
```
python3 pr_tree_main.py -x dataset/Boston.csv -sg 1.25,...,1
```
This will give the same average and std without the need to go through the validation process.

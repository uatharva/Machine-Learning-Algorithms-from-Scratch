# Machine-Learning-Algorithms-from-Scratch
## Perceptron Learning Algorithm
On the linear separable dataset the perceptron algorithm runs smoothly by producing the accuracy of 100%. After 10 fold validation the accuracy came to be 72% with the difference between the original accuracy and after validation to be 28%. The parameters used in this algorithm are the weights which are updated at each iteration when the output vector is not equal to the target or label vector and along with that the learning rate, which is in this case is kept to be 0.01. While in the breast cancer dataset the algorithm runs perfectly with the accuracy of 80%. While after validation of 10 folds the accuracy came to be 69% with the difference of 11% between the original accuracy and the accuracy after 10 fold validation. 
The termination of the linear separable dataset is normal because the perceptron algorithms work fine only when the data is linearly separable. By the name linear separable means- it can be classified linearly. This is because perceptron is a linear classifier i.e it will never get to the state with all the input vectors classified correctly if the training set D is not linearly separable, i.e. if the positive examples cannot be separated from the negative examples by a hyperplane. If the training set is linearly separable, then the perceptron is guaranteed to converge. 
While the breast cancer dataset was non-linear. That’s why it never converges. For it to converge we stop it after some iterations and that depends on the size of the dataset. In our dataset we have 5 feature vectors and for such complex data with more than 2 features it is impossible for perceptron to classify the features into 2 linear halfspaces which is the main goal of a perceptron. And because of that it never converges and hence we get errors. After ERM on this data we get the accuracy of 80%. And after 10 fold cross validation we get 69%, i.e 31% error. This error occurs only because the perceptron cannot find a way to classify all the features into binary categories and hence never converges. This does not happen with the linearly separable dataset because in that we have only 2 feature vectors and hence the perceptron can easily classify the features into its respective classes as the work of a classifier. 
## Support Vector Machine Algorithm 
SVM with maqximizing margin separating hyperplane that runs on 20% of the test set.
## K Nearest Neighbors Algorithm
As the data is randomized, therefore the accuracy is changing everytime the code is executed for any value of k. 
## K Means Clutering Algorithm 
When k =2, i.e when 2 clusters are made, then the algorithm is able to separate the
patients based on the label feature ‘diagnosis’ up to approximately 80% of the
time. Two clusters are being made - one of 1 label(1 being diagnosed as cancer)
and other of 0 label(0 being not diagnosed as cancer). In the cluster with label 1
total diagnosis count was : 444 and out of that 354 were predicted to be 1 and other
90 to be 0, i.e the positive diagnosis was 79.72%. While in the cluster with label 0 :
the total diagnosis count was 124, out of which 122 were diagnosed 0 and 2 1
labels were incorrectly clustered into that cluster making the accuracy of the
negative diagnosis to be 98.4%. Although the data was randomized, results were
the same for both the distances same at each run.

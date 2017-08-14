# phd-thesis-iii
Simple Agenda as follows

## __Implement mRMR to reduce dataset - done (JMIM, JMI, mRMR usinc scikit)__

## Implement Treelet Clustering on reduced dataset

## Implement SVM | LDA | ANT | ANT Miner | BAT | BAT Miner on the reduced dataset - __ PSO not yet applied for Feature Selection, classifiers used so far are SVM | ExtraTree | RF | MLP | AdaBoost and Decision Tree __

## Use reliability based error measure for training and validation 

## __Implement Ensemble - done __

## Produce Results and Fine Tune - __ Fine Tuning still in progress__

# Progress Tracking
## Oct 2016
* Planned out the strategy to implement the agenda/strategy outlined above.
* Tried to implement mrmr from https://github.com/nlhepler/mrmr but for some reason it seemed to require lots of updates, lots of data type conversions and still the end results it produced for me were not reliable as it seemed to generate sequential list of consecutive vectors from the left of the matrix.

## Nov 2016
* Finally found https://github.com/danielhomola/mifs as an embarrasingly parallel implementation of mrmr and it worked for me.
* Now implementing Treelet & ANT with the produced subset

## Dec 2016
* Started using Python's Scikit to start producing results on CPUs (using multiple cores via python). If needed we will use GPU again.
* Implemented Email notifications to get notified when long running tasks get completed.
* Explored and used mRMR, JMI and JMIM based Feature Selection
* Explored and shortlisted SVM, ExtraTree, RandomForest, MLP, AdaBoost and Decision Trees
* Started listing down the results in google sheet to be able to identify trends or patterns
* Keeping trained classifiers and datasets as pickles so we could reload them when needed and move ahead with Ensemble creation

## Jan 2017
### replanned the following pathway
* Applying FS multiple times to reduce from several thousand features to 250 features and then from 250 to 200, 150, 100, 50 and 10 best features to see what works best for us
* Exploring to see if GA can be used to tune parameters for the classifiers - not exactly sure how to use it though

Replan to do the following tasks along the way
1. Retrain the models for other datasets - so far we have done so only on 1 dataset (DataSet A)
2. Decide on Ensemble Scheme i.e. simple voting or weighted avg or even SVN/ANN on individual classifiers
3. Calculate overall Ensemble performance
4. Use LOOCV instead of 10 CV
5. Try out different forms of error functions, including Mattia's reliability parameter
6. Enlist all parameter values for each classifier for each settings and decide on which parameters to tune in what range.
7. See if treelet/meta genes on top of mrmr can improve further
8. see if bagging/boosting can further help

## Mar 2017
Ran into issues when applying Feature Selection due to missing values in the columns. Wrote program to find out the reason for crash and realized this. Suggested approaches for now are...
1) remove such features altogether
2) impute values using avg of other vectors so it doesnt impact the selection based on this imputed value
3) impute values using a novel technique, find most correlated vector considering all available features and copy over the value from the most similar vector

Also figuring out the way to find best parameter values for the classifiers

## Jun 2017
* Started off with Dataset B and also restarted with Dataset A to include both normalized and unnormalized data
* Figuring out which of LOOCV and K-fold CV is best suited for us. As per Mattia's thesis, LOOCV is the best but turns out it results in worse accuracy in our case.
* Dumping all results in unformatted text files, this will be a challenge later to structure the data
* The individual results are not very encouraging in terms of accuracy. Though the time it takes is pretty affordable now.
* Also started creating Ensembles to see if that helps in getting better results

## Jul 2017
* Even Ensembles are not proving to be very helpful with better accuracy. Seems like we need to revisit our strategy.
* Grouped the generated results into folders
* Writing parser to parse and extact results
* Dumping the data into a database

## Aug 2017
* Repopulated DB from the results files
* Preparing the results for meeting with Advisor

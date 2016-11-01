# phd-thesis-iii
Simple Agenda as follows

## Implement mRMR to reduce dataset

## Implement Treelet Clustering on reduced dataset

## Implement SVM | LDA | ANT | ANT Miner | BAT | BAT Miner on the reduced dataset

## Use reliability based error measure for training and validation 

## Implement Ensemble

## Produce Results and Fine Tune

# Progress Tracking
## Oct 2016
* Planned out the strategy to implement the agenda/strategy outlined above.
* Tried to implement mrmr from https://github.com/nlhepler/mrmr but for some reason it seemed to require lots of updates, lots of data type conversions and still the end results it produced for me were not reliable as it seemed to generate sequential list of consecutive vectors from the left of the matrix.

## Nov 2016
* Finally found https://github.com/danielhomola/mifs as an embarrasingly parallel implementation of mrmr and it worked for me.
* Now implementing Treelet & ANT with the produced subset

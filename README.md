# Song_Classifier

# Musical Genre Classifcation and Feature Analysis of Country and Pop Lyrics

## Overview

This project was created as a part of the CS229 Machine Learning course at Stanford University. 

This project was inspired by the apparent importance of certain key words (i.e. "truck", "beer", "girl")
in genre classification, especially in distinguishing between pop and country songs. The aim of the project was to
accurately classify pop and country songs based on lyrical content and to extract the most relevant key words that 
distinguished the two genres. 

This aim was achieved in the following steps: 
1. Data collection: from the Genius website using the genius API. Five songs were collected for each of twenty top country and pop artists.
2. Cleaning lyrics and tokenizing into words, removing stop words, representing words in bag of words and term frequency-inverse document frequency models. 
3. Applying a Naive Bayes Classifier, linear support vector machine, random forest classifier, and K-means clustering to training data. 
4. Extracting the top ten features for pop and country music from each model 
5. Evaluating models with 10-fold cross validation. 

### Results 
Model accuracy ranged from 0.74 to 0.85  ranged from 0.74 to 0.85 with the random forest classifier with TF-IDF performing best and the linear SVM with bag of words
performing the worst. 
Across all models, the word "little" stands out as the only word that appears consistently as a unique classifier of country songs. 
The word "whiskey" also appeared in multiple models as a key feature in country song lyrics. 

### Future Directions: 
The project may be improved by implementation with a more extensive data set or using a data set that corresponds top country songs rather than limiting song
selection to 20 artists. 

I also plan on extending the project in the future by implementing a neural network to generate a country song. 


### How to Use
- "lyrics.py" accesses the genius API and generates a raw lyrics dataset 
- "data.py" creates a train and test data set 
- "classifier.py" vectorizes lyrics, finds most frequent words for pop and country songs, runs and evaluates all models


## References
1. Smith, G. [Grady Smith]. Every country song has these lyrics. Right? Youtube. https://www.youtube.com/ watch?v=48ZxNFGJTo8
2. "How to Collect Song Lyrics with Python" https://towardsdatascience.com/song-lyrics-genius-api-dcc2819c29
3. Text Classification. (n.d.). Re- trieved November 5, 2021, from https://learning.oreilly.com/library/view/text- analytics-with/9781484243541/ html/4272872En5Chapter.xhtml

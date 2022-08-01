import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
                            f1_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.feature_extraction import text
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nltk.stem.snowball import EnglishStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.preprocessing import normalize
from os import path





def load_lyrics(lyrics_path):
    # loads lyrics csv into panda data frame and splits into train and test
    # lyrics_path. Path to csv file containing lyrics
    # returns X_train, x_test, y_train, y_test as panda dataframes
    # load lyrics dataset into panda dataframe
    df = pd.read_csv(lyrics_path, header=None)
    # drop "borderline" lyrics
    #df.drop[[210]]
    # split into labels (features) and labels
    lyrics = df[[1]]
    lyrics.columns = ['lyrics']
    labels = df[[2]]
    labels.columns = ['label']
    print(lyrics.head())
    print(labels.head())

    # split data set into train and test in 70/30 ratio
    X_train, X_test, y_train, y_test = train_test_split(lyrics, labels, test_size=0.3)
    # print(X_train.head())
    # print(X_test.head())
    print(y_train.head())
    print(y_test.head())

    return X_train, X_test, y_train, y_test

def model_performance(Model_name, y_test, y_pred):
    # takes in y and y_pred data sets and outputs measures of model accuracy
    #writes to file

    #generate and display confusion matrix:
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(Model_name + " Accuracy: ", accuracy)
    print(Model_name + " Precision: ", precision)
    print(Model_name + " Recall: ", recall)
    print(Model_name + " F1: ", f1)

    file1 = open("Model_Eval.txt", "a")
    file1.write(Model_name + "\n")
    file1.write(Model_name + " Accuracy: " + str(accuracy) + "\n")
    file1.write(Model_name + " Precision: " + str(precision) + "\n")
    file1.write(Model_name + " Recall: " + str(recall) + "\n")
    file1.write(Model_name + " F1: " + str(f1) + "\n")


def get_features(vectorizer, X_train, X_test):
    train_feat = vectorizer.fit_transform(X_train.lyrics)
    test_feat = vectorizer.transform(X_test.lyrics)
    return train_feat, test_feat

def predict(model, train_feat, y_train, test_feat):
    #fits given model and returns predicted labels
    model.fit(train_feat, y_train.label)
    y_pred = model.predict(test_feat)
    return y_pred

def run_model (X_train, X_test, y_train, y_test, model_name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_performance(model_name, y_test, y_pred)
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, train_pred)
    print("Train Accuracy: ", accuracy_train)

    # write output to file
    file1 = open("Model_Eval.txt", "a")
    file1.write(model_name + "\n")
    file1.write(str(scores.mean()) + " accuracy with a standard deviation of" + str(scores.std()) + " \n")
    file1.write("Train accuracy of " + str(accuracy_train))
    file1.close()

def get_salient_words(nb, vect, class_id):
    """ Returns salient words for a given class
    Parameters:
        nb: naive bayes classifier
        vect: count vectorizer
        class_id: class id (1 or 0)

    Returns:
        sorted list of words (word, log prob)

    Source: https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes
          """
    words = vect.get_feature_names()
    zipped = list(zip(words, nb.feature_log_prob_[class_id]))
    sorted_zip = sorted(zipped, key=lambda t: t[1], reverse=True)

    return sorted_zip
##defines tokenizer to split words based on spaces
def space_tokenizer(text):
    return re.split("\\s+", text)

def stemmed_words(doc):
    """defined stemmer for use in count vectorizer"""
    stemmer = EnglishStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))

def mut_inf_plot(model_name, model, vectorizer, save_path, X_train, y_train, stop_words):
    """ runs mutual infomration on a given model and plots model accuracy against number of features selected"""
    k_vec = [10, 50, 100, 200, 300, 400, 500, 550]
    accuracy_vals = []
    for k in k_vec:
        text_clf = Pipeline([('vect', vectorizer),
                             ('reducer', SelectKBest(mutual_info_classif, k=k)),
                             ('clf', model)])
        text_clf.fit(X_train, y_train)
        scores = cross_val_score(text_clf, X_train, y_train, cv=10)
        accuracy = scores.mean()
        accuracy_vals.append(accuracy)

    k_vals = np.array(k_vec)
    accuracy_vals = np.array(accuracy_vals)

    ### plot features
    plt.figure(figsize=(10, 8))
    plt.plot(k_vals, accuracy_vals, marker='.')
    plt.xscale('log')
    plt.xlabel('k features used')
    plt.minorticks_off()
    plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
    plt.ylabel('Accuracy')
    using_all = accuracy_vals[-1]
    plt.plot(k_vals, using_all * np.ones_like(k_vals), color='orange', label='Using all features')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def main(keywords=True, naive_bayes=True, svm_linear=True, random_forest=True, kmeans=True, results=True):

########## Load and Pre-process Data############################################

    #### load train and test set

    ###load country lyrics for word cloud/frequency creation
    df_train = pd.read_csv("lyrics_plus.csv", header=None)
    df_train.columns = ['artist', 'lyrics', 'label']
    country = df_train[df_train.label == 1]
    X_country = country.lyrics
    y_country = country.label

    #load pop lyrics for word cloud/frequency creation
    pop = df_train[df_train.label == 0]
    X_pop = pop.lyrics
    y_pop = pop.label

    #load train dataset
    df_train = pd.read_csv("lyrics_train", header=None)
    X_train = df_train[[0]]
    X_train.columns = ['lyrics']
    y_train = df_train[[1]]
    y_train.columns = ['label']

    #load test dataset
    df_test = pd.read_csv("lyrics_test", header=None)
    X_test = df_test[[0]]
    X_test.columns= ['lyrics']
    y_test = df_test[[1]]
    y_test.columns = ['label']

    y_train = y_train.to_numpy()
    y_train = y_train.reshape((y_train.shape[0], ))



    ##create new stop words based on results from vectorizer
    new_stop_words = ["i'm", "like", "just", "yeah", "don't", "got", "it's", "you're", "oh", "can't", "i'll",
                      "she's", "ooh", "i've", "know", "'cause", "let", "that's", "said", "say", "let's", "i'd","there's",
                      "won't", "come", "gotta", "gonna", "tell", "cause"]
    stop_words = text.ENGLISH_STOP_WORDS.union(new_stop_words)

    ## bag of words vectorizer
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=5, tokenizer=space_tokenizer)
    train_feat = vectorizer.fit_transform(X_train.lyrics)
    test_feat = vectorizer.transform(X_test.lyrics)
    feature_array = vectorizer.get_feature_names()

    ### tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=space_tokenizer)
    tfidf_feat = tfidf_vectorizer.fit_transform(X_train.lyrics)
    tfidf_test_feat = tfidf_vectorizer.transform(X_test.lyrics)
    tfidf_feature_array = tfidf_vectorizer.get_feature_names()


######################Find Key Words ###################################################
    if keywords == True:
        # #### find key words
        ##create new stop words based on results from vectorizer
        new_stop_words = ["i'm", "like", "just", "yeah", "don't", "got", "it's", "you're", "oh", "can't", "i'll", "she's",
                          "ooh", "i've", "know", "'cause", "let", "that's", "said", "say", "let's", "i'd", "there's", "won't",
                          "come", "gotta", "gonna", "tell", "cause"]
        stop_words = text.ENGLISH_STOP_WORDS.union(new_stop_words)

        #create vectorizer/features for country songs
        # country_vect = CountVectorizer(stop_words=stop_words, min_df=5, tokenizer=space_tokenizer)
        # country_feat = country_vect.fit_transform(X_country)

        country_vect= TfidfVectorizer(stop_words=stop_words, min_df=5, tokenizer=space_tokenizer)
        country_feat = country_vect.fit_transform(X_country)

        top_n = 50
        #print top_n most frequent words
        print('Frequency: \n', sorted(list(zip(country_vect.get_feature_names(),
                                               country_feat.sum(0).getA1())),
                                      key=lambda x: x[1], reverse=True)[:top_n])
        ##create vectorizer/features for pop songs and print top_n words
        pop_vect = CountVectorizer(stop_words=stop_words, min_df=5, tokenizer=space_tokenizer)
        pop_feat = pop_vect.fit_transform(X_pop)
        print('Frequency: \n', sorted(list(zip(pop_vect.get_feature_names(),
                                               pop_feat.sum(0).getA1())),
                                      key=lambda x: x[1], reverse=True)[:top_n])

        ###Visualize Most Common Words for Pop/Country w/ word cloud###########
        # pop = ' '.join(X_pop)
        # wordcloud_pop = WordCloud(stopwords=stop_words, collocations=True).generate(pop)
        # plt.imshow(wordcloud_pop, interpolation='bilInear')
        # plt.axis('off')
        # plt.show()
        #
        # country = ' '.join(X_country)
        # wordcloud_country = WordCloud(stopwords=stop_words, collocations=True).generate(country)
        # plt.imshow(wordcloud_country, interpolation='bilInear')
        # plt.axis('off')
        # plt.show()

######################NAIVE BAYES IMPLEMENTATION ###############################################################
    if naive_bayes==True:

    # ###train Naive Bayes on BOW Vectorizer and get salient features
        print("Naive Bayes Model w/ BOW")
        nb_bow = MultinomialNB()

        run_model(train_feat, test_feat, y_train, y_test, "Naive_Bayes_BOW", nb_bow)
        pop_salient_bow = get_salient_words(nb_bow, vectorizer, 0)[:10]
        country_salient_bow = get_salient_words(nb_bow, vectorizer, 1)[:10]
        print(pop_salient_bow)
        print(country_salient_bow)

        print("Naive Bayes Model w/ TFIDF")
    ##train Naive Bayes on TFIDF Vectorizer and get salient features
        nb_tf = MultinomialNB()
        run_model(tfidf_feat, tfidf_test_feat, y_train, y_test, "Naive_Bayes_TFIDF", nb_tf)
        pop_salient_tfidf = get_salient_words(nb_tf, tfidf_vectorizer, 0)[:10]
        country_salient_tfidf = get_salient_words(nb_tf, tfidf_vectorizer, 1)[:10]
        print(pop_salient_tfidf)
        print(country_salient_tfidf)

    ##run mutual information for naive bayes w/count vectorizer and extract key features
        print("Naive Bayes Model w/ Reduced Features")
        nb_red = MultinomialNB()
        mut_inf_plot("Naive Bayes", nb_red, vectorizer, "accuracy_plot_NB.png", X_train.lyrics, y_train, stop_words)
        #use 500 features based on results from mutual_information_plot
        text_clf = Pipeline([('vect', vectorizer),
                             ('reducer', SelectKBest(mutual_info_classif, k=500)),
                             ('clf', nb_red)])
        text_clf.fit(X_train.lyrics, y_train)
        scores = cross_val_score(text_clf, X_train.lyrics, y_train, cv=10)
        accuracy = scores.mean()
        print("Accuracy: ", accuracy)
        print("Std Dev: ", scores.std())


        train_pred = text_clf.predict(X_train.lyrics)
        accuracy_train = accuracy_score(y_train, train_pred)
        print("Train Accuracy", accuracy_train)


        pop_salient_red = get_salient_words(text_clf['clf'], text_clf['vect'], 0)[:10]
        country_salient_red = get_salient_words(text_clf['clf'], text_clf['vect'], 1)[:10]
        print(pop_salient_red)
        print(country_salient_red)

        ##run mutual information for naive bayes w/tf-idf vectorizer and extract key features
        nb_red_tf = MultinomialNB()
        mut_inf_plot("Naive Bayes", nb_red_tf, tfidf_vectorizer, "accuracy_plot_NB_tfidf.png", X_train.lyrics, y_train, stop_words)
        ##use all features

    #######Linear SVM Model###########################################################################################
    if svm_linear:

    #### run svm model with BOW model
        print("Linear SVM model using all features and BOW")
        ### find optimal c-value
        c_vec = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        accuracy = []
        train_accuracy = []
        for c in c_vec:
            lin_svm = svm.SVC(kernel='linear', C=c)
            lin_svm.fit(train_feat, y_train)
            scores = cross_val_score(lin_svm, train_feat, y_train, cv=10)
            accuracy.append(scores.mean())
            train_pred = lin_svm.predict(train_feat)
            train_accuracy.append(accuracy_score(y_train, train_pred))

        c_vec = np.array(c_vec)
        accuracy = np.array(accuracy)
        train_accuracy = np.array(train_accuracy)
        ### plot learning curve for c

        plt.figure(figsize=(10, 8))
        plt.plot(c_vec, accuracy, marker='.', label="Test Accuracy")
        plt.plot(c_vec, train_accuracy, marker='o', label="Train Accuracy")
        plt.xscale('log')
        plt.xlabel('C-values')
        plt.minorticks_off()
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("Linear SVM Learning Curve")
        plt.show()

        ### c value of 0.01 obtains best accuracy
        lin_svm = LinearSVC(C=0.01)
        run_model(train_feat, test_feat, y_train, y_test, "SVM_BOW", lin_svm)

        ##run mutual information

        mut_inf_plot("Linear SVM", lin_svm, vectorizer, "accuracy_plot_SVM.png", X_train.lyrics, y_train, stop_words)
        # ###using tfidf

        print("Linear SVM model using all features and TFIDF")
        ## find optimal c-value
        c_vec = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                 0.8, 0.9, 1.0]
        accuracy = []
        train_accuracy = []
        for c in c_vec:
            lin_svm_tf = LinearSVC(C=c)
            lin_svm_tf.fit(tfidf_feat, y_train)
            scores = cross_val_score(lin_svm_tf, tfidf_feat, y_train, cv=10)
            accuracy.append(scores.mean())
            train_pred = lin_svm_tf.predict(tfidf_feat)
            train_accuracy.append(accuracy_score(y_train, train_pred))

        c_vec = np.array(c_vec)
        accuracy = np.array(accuracy)
        train_accuracy = np.array(train_accuracy)
        ### plot learning curve for c

        plt.figure(figsize=(10, 8))
        plt.plot(c_vec, accuracy, marker='.', label="Test Accuracy")
        plt.plot(c_vec, train_accuracy, marker='o', label="Train Accuracy")
        plt.xscale('log')
        plt.xlabel('C-values')
        plt.minorticks_off()
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("Linear SVM_TF-IDF Learning Curve")
        plt.show()


        ###use C=0.8
        lin_svm_tf = LinearSVC(C=0.8)
        run_model(tfidf_feat, tfidf_test_feat, y_train, y_test, "SVM_TFIDF", lin_svm_tf)

    ##get top features from BOW model
        top_features = 10
        feature_names = vectorizer.get_feature_names()
        coef = np.array(lin_svm.coef_).ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()

    ## get top features from TF-IDF model
        top_features = 10
        feature_names = tfidf_vectorizer.get_feature_names()
        coef = np.array(lin_svm_tf.coef_).ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()
#####______________________________________________________________________________________________________

    #### implement svm with rbf kernel

    #mut_inf_plot("RBF SVM", rbf_svm, "accuracy_plot_SVM_rbf.png", train_feat, y_train)
    ### use 500 features

    ### select c and gamma
    # mi_features = SelectKBest(mutual_info_classif, k=100)
    # feat_small = mi_features.fit_transform(train_feat, y_train)
    # test_small = mi_features.transform(test_feat)

    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    # grid.fit(feat_small, y_train)
    #
    # print(
    #     "The best parameters are %s with a score of %0.2f"
    #     % (grid.best_params_, grid.best_score_)
    # )

    # #fit SVM with c = 100 and gamma = 0.0001
    # rbf_svm = svm.SVC(C=100, gamma=0.0001)
    # run_model(feat_small, test_small, y_train, y_test, "RBF SVM w/ 100 feat", rbf_svm)

############________________________________________________________________________________________
    if random_forest:
        ##implement random forest classifier

        ### find optimal max_depth-value
        depth_vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15]
        accuracy = []
        train_accuracy = []
        for depth in depth_vec:
            rfc = RandomForestClassifier(max_depth=depth)
            rfc.fit(train_feat, y_train)
            scores = cross_val_score(rfc, train_feat, y_train, cv=10)
            accuracy.append(scores.mean())
            train_pred = rfc.predict(train_feat)
            train_accuracy.append(accuracy_score(y_train, train_pred))

        c_vec = np.array(depth_vec)
        accuracy = np.array(accuracy)
        train_accuracy = np.array(train_accuracy)
        ### plot learning curve for c

        plt.figure(figsize=(10, 8))
        plt.plot(c_vec, accuracy, marker='.', label="Test Accuracy")
        plt.plot(c_vec, train_accuracy, marker='o', label="Train Accuracy")
        plt.xscale('log')
        plt.xlabel('Max_depth-values')
        plt.minorticks_off()
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("Random Forest Classifier Learning Curve")
        plt.show()

        ##use 10 for max_depth
        rfc = RandomForestClassifier(max_depth=10)
        rfc_tf = RandomForestClassifier(max_depth=10)
        #mut_inf_plot("Random Forest", rfc, vectorizer, "accuracy_plot_random_forest.png", X_train.lyrics, y_train, stop_words)
        ###decide to use all features
        run_model(train_feat, test_feat, y_train, y_test, "Random Forest, md=10", rfc)
        run_model(tfidf_feat, tfidf_test_feat, y_train, y_test, "Random Forest, md=10, tfidf", rfc_tf)
        #get top features for BOW model
        imp_feat = rfc.feature_importances_
        indices = np.argsort(imp_feat)[::-1]
        feature_names = vectorizer.get_feature_names()
        top_words = []
        for i in xrange(10):
            top_words.append(feature_names[indices[i]])
        print(top_words)

        #get top features for TFIDF model
        imp_feat = rfc_tf.feature_importances_
        indices = np.argsort(imp_feat)[::-1]
        feature_names = tfidf_vectorizer.get_feature_names()
        top_words = []
        for i in xrange(10):
            top_words.append(feature_names[indices[i]])
        print(top_words)
        ## create decision tree for random forest classifier
        estimator = rfc_tf.estimators_[5]
        # Export as dot file
        export_graphviz(estimator,
                        out_file='tree.dot',
                        feature_names=feature_names,
                        class_names=['pop', 'country'],
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
        img = mpimg.imread('tree.png')
        imgplot = plt.imshow(img)
        plt.show()

    if kmeans:
        ### normalize features
        train_feat_norm = normalize(train_feat)
        train_feat_array = train_feat_norm.toarray()

        ###use PCA to reduce components to visualize kmeans
        pca = PCA(n_components= 2)
        PCA_feat = pca.fit_transform(train_feat_array)
        k_model = KMeans(n_clusters=2)
        k_model.fit(PCA_feat)
        pred_val = k_model.predict(PCA_feat)
        plt.scatter(PCA_feat[:,0], PCA_feat[:,1], c=pred_val, s=50, cmap='viridis')
        plt.show()

        ## k means w/o PCA BOW
        model =KMeans(n_clusters=2)
        model.fit(train_feat)

        model_tf = KMeans(n_clusters=2)
        model_tf.fit(tfidf_feat)

        #### print top words per class
        print("Top Words BOW")
        order_centroids= model.cluster_centers_.argsort()[:, ::-1]
        terms=vectorizer.get_feature_names()
        for i in range(2):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print( ' %s' % terms[ind]),
            print()

        #### print top words per class
        print("Top Words TFIDF")
        order_centroids = model_tf.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf_vectorizer.get_feature_names()
        for i in range(2):
            print("Cluster %d:" % i),
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind]),
            print()

        k_labels_red = k_model.labels_
        k_labels = model.labels_
        distance_0 = model.transform(train_feat)[:,0]
        distance_1 = model.transform(train_feat)[:,1]

        k_classification = np.column_stack((X_train, k_labels_red, k_labels, distance_0, distance_1))
        k_classification = pd.DataFrame(k_classification)
        k_classification.columns = ['lyrics', 'reduced', 'BOW', "distance_0", "distance_1"]
        print(k_classification[k_classification['reduced']== 0])
        print(k_classification[k_classification['BOW'] == 0])
        sorted_0 =k_classification.sort_values(by=['distance_0'])
        np.savetxt('k_means_labeled_0.csv', sorted_0, fmt=('%s'), delimiter=',')

        sorted_1 = k_classification.sort_values(by=['distance_1'])
        np.savetxt('k_means_labeled_1.csv', sorted_1, fmt=('%s'), delimiter=',')

    if results:
        result_words = pd.read_csv("Result_Words.csv", nrows=60)
        print(result_words)
        pop_words = result_words['Pop']
        pop_words =' '.join(pop_words)
        country_words = result_words['Country']
        country_words =' '.join(country_words)

        wordcloud_pop = WordCloud(stopwords=stop_words, collocations=True, background_color='white', colormap='tab10').generate(pop_words)
        plt.imshow(wordcloud_pop, interpolation='bilInear')
        plt.axis('off')
        plt.show()

        wordcloud_country = WordCloud(stopwords=stop_words, collocations=True,background_color='white', colormap='tab10').generate(country_words)
        plt.imshow(wordcloud_country, interpolation='bilInear')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main(keywords=False, naive_bayes=False, svm_linear=False, random_forest=False, kmeans=True, results=False)


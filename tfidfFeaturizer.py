from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer

#This is a subclass that extends the abstract class Featurizer.
class tfidfFeaturizer(Featurizer):

    #The abstract method from the base class is implemeted here to return count features
    def getFeatureRepresentation(self, train, val):
        vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)
        return vectorizer.fit_transform(train), vectorizer.transform(val)

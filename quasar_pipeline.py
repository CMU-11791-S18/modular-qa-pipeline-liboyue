import sys
import json
from sklearn.externals import joblib

from Retrieval import Retrieval
from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
from tfidfFeaturizer import tfidfFeaturizer

from MultinomialNaiveBayes import MultinomialNaiveBayes
from svm import SVM

from perceptron import Perceptron

from Evaluator import Evaluator

N = 100


class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance
        self.evaluatorInstance = Evaluator()
        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        self.trainData['questions'] = self.trainData['questions'][0:N]
        
        trainfile.close()
        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()
        #self.question_answering()
        self.prepare_data()
        self.prepare_features()

    def makeXY(self, dataQuestions):
        X = []
        Y = []
        for question in dataQuestions:
            
            long_snippets = self.retrievalInstance.getLongSnippets(question)
            short_snippets = self.retrievalInstance.getShortSnippets(question)
            
            X.append(short_snippets)
            Y.append(question['answers'][0])
            
        return X, Y


    def get_data(self):
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates'] ##
        return self.makeXY(self.trainData['questions'])


    def prepare_data(self):
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates'] ##

        self.X_train, self.Y_train = self.makeXY(self.trainData['questions'])
        self.X_val, self.Y_val_true = self.makeXY(self.valData['questions'])

    def prepare_features(self):
        #featurization
        self.X_features_train, self.X_features_val = self.featurizerInstance.getFeatureRepresentation(self.X_train, self.X_val)

    def qa(self):
        self.clf = self.classifierInstance.buildClassifier(self.X_features_train, self.Y_train)
        #Prediction
        Y_val_pred = self.clf.predict(self.X_features_val)
        
        a = self.evaluatorInstance.getAccuracy(self.Y_val_true, Y_val_pred)
        p, r, f = self.evaluatorInstance.getPRF(self.Y_val_true, Y_val_pred)

        print("Accuracy: " + str(a))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F-measure: " + str(f))
        


#if __name__ == '__main__':
# from quasar_pipeline import *
#trainFilePath = sys.argv[1] #please give the path to your reformatted quasar-s json train file
#valFilePath = sys.argv[2] # provide the path to val file
#a = json.load(open('data/msmarco_train_formatted.json'))
#a['questions'][0].keys()

train_path = 'data/quasar-s_train_formatted.json'
val_path = 'data/quasar-s_dev_formatted.json'

for N in [5000, 7000, 10000]:
    print ('N = ' + str(N))
    print()

    p = Pipeline(train_path, val_path, Retrieval(), CountFeaturizer(), MultinomialNaiveBayes())
    print('Count MNB')
    p.qa()

p.classifierInstance = SVM()
print()
print('Count SVM')
p.qa()

p.classifierInstance = Perceptron()
print()
print('Count perceptron')
p.qa()


p.classifierInstance = MultinomialNaiveBayes()
p.featurizerInstance = tfidfFeaturizer()
p.prepare_features()
print()
print('TF-IDF SVM')
p.qa()

p.classifierInstance = MultinomialNaiveBayes()
print()
print('TF-IDF MNB')
p.qa()

p.classifierInstance = Perceptron()
print()
print('TF-IDF perceptron')
p.qa()


#p = Pipeline(train_path, val_path, Retrieval(), tfidfFeaturizer(), MultinomialNaiveBayes())


#x, y = p.get_data()
#x_f, y_f = featurizerInstance.getFeatureRepresentation(x, y)
#x_f, y_f = SVM().getFeatureRepresentation(x, y)



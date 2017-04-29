from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pickle
import numpy as np

def linefeatures(filename):
    try:
        picklefile = open('vocabdict.pickle','rb+')
        vocab_dict = pickle.load(picklefile)
        if vocab_dict != None:
            return vocab_dict
    except:
        pass
    
    stem = PorterStemmer()
    raw_text = open(filename).read()
    tokenized_words = word_tokenize(raw_text)
    unique_list = nltk.unique_list(tokenized_words)
    #print(unique_list)
    stop = set(stopwords.words('english'))
    new_list = []
    for word in unique_list:
        if(word.lower().isdigit()):
            continue
        word = stem.stem(word)
        if (word not in stop):
            new_list.append(word)
    unique_list = nltk.unique_list(new_list)
    i = 0
    vocab_dict = {}
    #print(unique_list)
    for word in unique_list:
       vocab_dict[word] = i
       #vocab_dict[word] = i;
       i+=1
    #print(vocab_dict)
    with open('vocabdict.pickle','wb+') as picklefile:
        pickle.dump(vocab_dict,picklefile)
    return vocab_dict


def getPredictX(vocab_dict):
    feature_len = len(vocab_dict)
    X,_ = readData('PredictData.txt')
    count=0
    numpy_X = np.empty([len(X),feature_len],dtype=np.float)
    for line in X:
        line_array = np.zeros([1,feature_len],dtype=np.float)
        for word in line:
            index = vocab_dict.get(word)
            if index!=None:
                line_array[0,index-1]=1;
        numpy_X[count] = line_array
        count+=1;
    return numpy_X;


def readData(filename):
    
    X = [[]]
    Y = []
    stop = set(stopwords.words('english'))
    stem = PorterStemmer()
    with open(filename) as file:
        for line in file:
            tokenized_line = word_tokenize(line)
            line_list = []
            for word in tokenized_line:
                word = stem.stem(word)
                if(word.isdigit()):
                    Y.append(int(word))
                    continue
                elif(word not in stop):
                    line_list.append(word)
                #print(line_list)
            line_list = list(nltk.unique_list(line_list))
            X.append(line_list)
            #print(line_list)
    X = X[2:]
    #print(len(X))
    Y = Y[1:]
    #print(len(Y))
    #print(Y)
    return X,Y

def createFeatureSet(feature_len,vocab_dict,read_data_function):
    try:
        fileX = open('X.pickle','rb+')
        fileY = open('Y.pickle','rb+')
        numpy_X = pickle.load(fileX)
        numpy_Y = pickle.load(fileY)
        if(numpy_X != None and numpy_Y != None):
            fileX.close()
            fileY.close()
            return numpy_X,numpy_Y
    except :
        pass
    X,Y = readData('trainingdata.txt')
    count=0
    numpy_X = np.empty([len(X),feature_len],dtype=np.float)
    for line in X:
        line_array = np.zeros([1,feature_len],dtype=np.float)
        for word in line:
            index = vocab_dict.get(word)
            if index!=None:
                line_array[0,index-1]=1;
        numpy_X[count] = line_array
        count+=1;
    #print(Y)
    #for category in Y:
    #    temp= np.array([1],dtype=np.float)
    #    temp[0] = category
    #    numpy_Y += temp
    #print (len(numpy_X),len(numpy_Y))
    numpy_Y = np.empty([len(Y),8],dtype=np.float)
    count=0
    for category in Y:
        temp = np.zeros([1,8],dtype=np.float)
        temp[0,category-1]=1
        numpy_Y[count]=temp
        count+=1
    #numpy_Y = np.transpose(numpy_Y)
    #print(numpy_Y)
    fileX = open('X.pickle','wb')
    fileY = open('Y.pickle','wb')
    pickle.dump(numpy_X,fileX)
    pickle.dump(numpy_Y,fileY)
    fileX.close()
    fileY.close()
    return numpy_X,numpy_Y


def get_X_Y():

    vocab_dict = linefeatures('trainingdata.txt')

    #X,Y = readData('trainingdata.txt')

    #print(len(vocab_dict),len(X),len(Y));

    X,Y=createFeatureSet(len(vocab_dict),vocab_dict,read_data_function=readData)

    import keras.utils as utils

    #Y=utils.to_categorical(Y,num_classes=9)
    #print(Y)
    return X,Y,vocab_dict

def getTrainingandValidation(X,Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2)
    return X_train,X_test,y_train,y_test

def generate_model(input_length_col):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import adam

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_length_col))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def main():
    X,Y,vocab_dict = get_X_Y()
    #X_predict = X[-1,:]
    #X = X[1:-2,:]
    #print(len(X))
    ##print(X_predict)
    X_train,X_test,y_train,y_test = getTrainingandValidation(X,Y)

    model = generate_model(len(X_train[1]))
    model.fit(X_train, y_train,
          epochs=10,
          batch_size=128)
    
    score = model.evaluate(X_test, y_test, batch_size=128)
    print(score)

    predict_X = getPredictX(vocab_dict)
    solns = model.predict(predict_X)
    maximum = solns.max(axis=1)
    for value in maximum:
        for i in range(0,len(solns)):
            for j in range(0,len(solns[0])):
                if(value==solns[i,j]):
                    print(j+1)
    
    
    
    
    

if __name__ == "__main__":
    import sys
    sys.exit(int(main() or 0))



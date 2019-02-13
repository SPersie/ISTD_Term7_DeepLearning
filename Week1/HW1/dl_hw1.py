import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

##### Read flower features from feats

def split_data(path):
    flowers = sorted(os.listdir(path))
    
    X_train = np.zeros((1, 512))
    X_val = np.zeros((1, 512))
    X_test = np.zeros((1, 512))
    X_final = np.zeros((1, 512))
    
    y_train = np.zeros(1)
    y_val = np.zeros(1)
    y_test = np.zeros(1)
    y_final = np.zeros(1)
    
    for i in range(int(len(flowers) / 80)):
        for j in range(0, 40):
            oneflower = np.load(path + flowers[i * 80 + j])
            re_oneflower = oneflower.reshape(1, 512)
            X_train = np.append(X_train, re_oneflower, axis=0)
            y_train = np.append(y_train, i + 1)
        
        for j in range(40, 60):
            oneflower = np.load(path + flowers[i * 80 + j])
            re_oneflower = oneflower.reshape(1, 512)
            X_val = np.append(X_val, re_oneflower, axis=0)
            y_val = np.append(y_val, i + 1)       
        
        for j in range(40, 60):
            oneflower = np.load(path + flowers[i * 80 + j])
            re_oneflower = oneflower.reshape(1, 512)
            X_test = np.append(X_test, re_oneflower, axis=0)
            y_test = np.append(y_test, i + 1)
            
        for j in range(0, 60):
            oneflower = np.load(path + flowers[i * 80 + j])
            re_oneflower = oneflower.reshape(1, 512)
            X_final = np.append(X_final, re_oneflower, axis=0)
            y_final = np.append(y_final, i + 1)
            
    y_train = y_train.reshape(681, 1)
    y_val = y_val.reshape(341, 1)
    y_test = y_test.reshape(341, 1)
    y_final = y_final.reshape(1021, 1)
    
    X_train = np.delete(X_train, (0), axis=0)
    X_val = np.delete(X_val, (0), axis=0)
    X_test = np.delete(X_test, (0), axis=0)
    X_final = np.delete(X_final, (0), axis=0)
    y_train = np.delete(y_train, (0), axis=0)
    y_val = np.delete(y_val, (0), axis=0)
    y_test = np.delete(y_test, (0), axis=0)
    y_final = np.delete(y_final, (0), axis=0)
    
    return X_train, X_val, X_test, X_final, y_train, y_val, y_test, y_final


def oneclasspredict(class_num, X_train, X_val, c):
    y_train = np.zeros((len(X_train),))
    y_val = np.zeros((340,))
    
    for i in range(int(len(X_train) / 17)):
        y_train[class_num * int(len(X_train) / 17) + i] = 1  

    for i in range(20):
        y_val[class_num * 20 + i] = 1
          
    
    svm = SVC(kernel='linear', C=c, probability=True)
    svm.fit(X_train, y_train)
    y_proba = svm.predict_proba(X_val)
    y_proba = np.delete(y_proba, (0), axis=1)
    
#    y_predict = (y_proba > 0.5) * 1
#    accuracy = accuracy_score(y_val, y_predict)
#    print(accuracy)
    return y_proba

def allclasspredict(X_train, X_val, y_val, c):
    predict_val = np.zeros((340, 2))
    for i in range(0, 17):
        oneclass = oneclasspredict(i, X_train, X_val, c)
        for j in range(0, 340):
            if oneclass[j][0] > predict_val[j][1]:
                predict_val[j][1] = oneclass[j][0]
                predict_val[j][0] = i + 1
    predict_val = np.delete(predict_val, (1), axis=1)
#    print(predict_val.shape)
#    accuracy = accuracy_score(predict_val, y_val)
    
    accuracy_list = []
    for i in range(0, 17):
        oneclass_val = predict_val[i * 20:(i + 1) * 20,]
        temp_val = np.full((20,), i + 1)
        oneclass_acc = accuracy_score(oneclass_val, temp_val)
        accuracy_list.append(oneclass_acc)
    
    accuracy = sum(accuracy_list) / len(accuracy_list)

    return predict_val, accuracy    

def selectC(X_train, X_val, y_val):
    best_c = 0
    best_val = None
    best_accuracy = 0
    c_list = [0.01, 0.1, 0.1 **0.5, 1, 10 **0.5, 10, 100 **0.5]
    
    for c in c_list:
        predict_val, accuracy = allclasspredict(X_train, X_val, y_val, c)
        if accuracy > best_accuracy:
            best_c = c
            best_val = predict_val
            best_accuracy = accuracy
            
    return best_c, best_val, best_accuracy
        

if __name__ == '__main__':
    ### Split the data set
    print('Start spliting data set.')
    path = '../flowers17_features/feats/'
    X_train, X_val, X_test, X_final, y_train, y_val, y_test, y_final = split_data(path)
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    print('Finish spliting data set.')
    
    ### Select the best regularization constant
    print('Start training to select regularization constant.')
    best_c, best_val, best_accuracy = selectC(X_train, X_val, y_val)
    print('The best validation accuracy is ', best_accuracy, 
          '. And it leads to the slection of C, which is ', best_c, '.')
    
    ### Train on the best C and predict test set    
    final_predict, final_acc = allclasspredict(X_final, X_test, y_test, best_c)
    print('The final accuarcy with selected C is ', final_acc, '.')
    
    

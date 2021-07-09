import numpy as np
import DATA_GEN

def pred_class(x, theta):
    "Separates data into classes using predicted theta"
    pred_y = DATA_GEN.prob(x, theta.reshape(2)) # probability of being in class 0
    class0 = []
    class1 = []
    y = []
    
    for i in range(len(pred_y)):
        if(pred_y[i] <= 0.5):
            class0.append(x[i])
            y.append(0)
        elif(pred_y[i] > 0.5):
            class1.append(x[i])
            y.append(1)
            
    return np.asarray(class0), np.asarray(class1), np.asarray(y)

def class_acc(y, y_pred):
    correct_class = 0
    for i in range(len(y)):
        if (y[i] == y_pred[i]):
            correct_class += 1
    return correct_class/len(y)
 
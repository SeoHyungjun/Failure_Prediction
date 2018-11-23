import sklearn.svm

def make_SVM(C = 1, logGamma = -3 ):
    
    return sklearn.svm.SVC(C=C, gamma=10**logGamma)
    

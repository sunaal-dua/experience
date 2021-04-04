from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict


def evaluation_metrics(ytrue, ypred_proba, multiclass, avg):
  '''
  Prints Accuracy, F1-Score and ROC-AUC score using Sklearn. Since we have multiclass classification you will need to specify 
  multi-class strategy and averaging strategy.
  NOTE THAT: 
  - ROC-AUC only support macro averaging; while f1-score supports both macro and micro
  - F1-score does not have multi-class parameter hence it does not take 'ovo' or 'ovr' into account. ROC-AUC supports multiclass
  - Accuracy is independent of these two parameters. They are not applicable to accuracy

  ytrue: true labels (array/list/series)
  ypred_proba: predicted probabilities (array/list/series)
  multiclass: multi-class strategy; enter 'ovr' for OneVsRest and 'ovo' for OneVsOne (string)
  avg: averaging strategy; accepts 'micro', 'macro' and 'weighted' (string)
  '''
  print("Accuracy:",accuracy_score(ytrue, ypred_proba.argmax(axis=1)))
  print("(macro) ROC-AUC:",roc_auc_score(y_true=ytrue, y_score=ypred_proba, multi_class=multiclass, average="macro"))
  print("("+avg+") F1 Score:",f1_score(y_true=ytrue, y_pred=ypred_proba.argmax(axis=1), average=avg))
  
  


def findingparameters_with_randomsearch(model,xtr,ytr,parameters,iterations, c_v=5, rs=1):
  '''
  Implements Sklearn's RandomizedSearchCV. By default 'scoring' is Accuracy since randomized search doesn't
  support ROC_AUC or 'ovo'/'ovr' ROC_AUC for multi-class labels

  model: pass your model/estimator
  xtr: training set features 
  ytr: training set labels
  parameters: model hyper-parameters that will be tuned (dict or list of dicts)
  iterations: number of parameter settings that are sampled (integer) 
  c_v: cross-validation generator (default = 5, integer)
  rs: Short for random_state. (default = 1, integer)

  Returns best k-cv accuracy score and corresponding parameters
  '''
  print("Model: {}\nNo. iterations for RandomSearch: {}\nCvFolds: {}\nRandom_State: {}".format(type(model),iterations,c_v,rs))
  print("Random Search CV started...")
  randomCV = RandomizedSearchCV(model, param_distributions=parameters, n_iter=iterations, cv=c_v, random_state=rs, scoring="accuracy")
  randomCV.fit(xtr, ytr)
  best = randomCV.best_params_
  print("\n\nBest {} fold cv accuracy score: {}\nBest parameters obtained:\n{}".format(c_v, randomCV.best_score_, best))
  return best



def tuned_CVperformance(model,x_tr,y_tr,x_te,y_te,c_v=5,svm=False):
  '''
  Returns the following performance metrics. Note that if svm = True; 3rd and 4th metrics are not evaluated
  because we are using SVM classifier wrapped with calibrated classifier which has its own cv capabilities.
  Hence when using SVM or Calibrated SVM please make sure to set svm = True to avoid errors.

  1) Gives OneVsOne ROC_AUC with default 'macro' averaging (best suited for skewed target distribution) 
  2) Gives accuracy
  3) Gives mean of k-fold cross validated OneVsOne ROC_AUC on test set
  4) Gives mean of k-fold cross validated accuracy on test set

  model: pass your model/estimator
  xtr: training set features 
  ytr: training set labels
  xte: testing set features 
  yte: testing set labels
  c_v: cross-validation generator (default = 5, integer)
  svm: If using SVM or Calibrated SVM please make sure to set this field to True to avoid errors (default=False)
  '''
  print("Fitting on the training set...")
  model.fit(x_tr,y_tr)

  print("Predicting probabilities for the training set...")
  tr_proba = model.predict_proba(x_tr)
  print("Training set OneVsOne AUC: ",roc_auc_score(y_tr,tr_proba,multi_class="ovo",average="macro"))
  print("Training set accuracy: ",accuracy_score(y_tr,tr_proba.argmax(axis=1)))

  print("\nPredicting probabilities for the testing set...")
  te_proba = model.predict_proba(x_te)
  print("Testing set OneVsOne AUC: ",roc_auc_score(y_te,te_proba,multi_class="ovo",average="macro"))
  print("Testing set accuracy: ",accuracy_score(y_te,te_proba.argmax(axis=1)))

  if(svm==False):
    print("\n"+str(c_v)+"-fold test set metrics:")
    pred_proba = cross_val_predict(model, x_te, y_te, cv=c_v, method='predict_proba')
    print("Test set mean "+str(c_v)+"-fold OneVsOne AUC:")
    print(roc_auc_score(y_te,pred_proba,multi_class="ovo",average="macro"))
    print("Test set mean "+str(c_v)+"-Fold CV Accuracy score:")
    print(accuracy_score(y_te,pred_proba.argmax(axis=1)))
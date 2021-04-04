import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


# function for training
def train_model(model,xtr,ytr,xte,yte,ep,optm="adam",bs=32,learn_rate=0.001,forTuning=False,callbacks=False,earlystop_params=None,reducelr_params=None,modelchkpnt_params=None):
  '''
  Trains models in different ways. You can use this for tuning the deep learning model as well as simple
  training. You can turn on/off callbacks and verbose as per your requirement.

  model: pass your model instance

  xtr: train set
  
  ytr: train labels
  
  xte: test set
  
  yte: test labels
  
  optm: optimizer. Takes string argument. Currently supports Adam (specify 'adam') and RmsProp (specify 'rmsprop')
  
  bs: batch size (default = 32)
  
  learn_rate: learning rate for the optimizer (default = 0.001)
  
  forTuning: when this is true it simply trains the model on the training set and evaluate on both train and test set. 
  It turns off verbose so you can try different learning rates in a loop. Returns trained model, train set metrics
  (accuracy and loss) and test set metrics (accuracy and loss) (Default=False)
  
  callbacks: when this is true it implements model checkpoint, early stopping and ReduceLRPlateau. If this is true,
  the next 3 parameters are compulsory (Default=False)

    earlystop_params: early stopping parameters (dictionary)
    eg: {'monitor':'val_loss', 'min_delta':0.1, 'patience':20, 'verbose':1, 'mode':'min', 'restore_best_weights':True}

    reducelr_params: ReduceLROnPlateau parameters (dictionary)
    eg: {'monitor':'val_loss', "factor":0.1, 'patience':20, 'verbose':1, 'mode':'min', 'min_delta':0.1, 'min_lr':1e-7}

    modelchkpnt_params: ModelCheckpoint parameters (dictionary) 
    eg: {"chkpnt_filename":"somefilename", "monitor":'val_accuracy', "verbose":1, "save_best_only":True, "mode":'max'}

  Returns trained model
  '''

  if(optm=="rmsprop"):
    opt = optimizers.RMSprop(learning_rate = learn_rate)
  elif(optm=="adam"):
    opt = optimizers.Adam(learning_rate = learn_rate)

  model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

  if(forTuning==True):
    model.fit(xtr, ytr, epochs=ep, batch_size=bs, validation_data=None, verbose=0)
    print("model has been fitted.. evaluating on train and test set")
    score_train = model.evaluate(xtr, ytr, batch_size=bs, verbose=0)
    score_test = model.evaluate(xte, yte, batch_size=bs, verbose=0)
    return [model,score_train,score_test]
  else:
    if(callbacks==True):
      chkpnt_path = "/content/drive/My Drive/Automatic Ticket Assignment/Model/model checkpoints/"+modelchkpnt_params["chkpnt_filename"]+".h5"
      checkpoint = ModelCheckpoint(chkpnt_path, monitor=modelchkpnt_params['monitor'], verbose=modelchkpnt_params["verbose"], save_best_only=modelchkpnt_params["save_best_only"], mode=modelchkpnt_params['mode'])
      stop = EarlyStopping(monitor=earlystop_params['monitor'], min_delta=earlystop_params['min_delta'], patience=earlystop_params['patience'], verbose=earlystop_params['verbose'], mode=earlystop_params['mode'], restore_best_weights=earlystop_params['restore_best_weights'])
      change_lr = ReduceLROnPlateau(monitor=reducelr_params['monitor'], factor=reducelr_params['factor'], patience=reducelr_params['patience'], verbose=reducelr_params['verbose'], mode=reducelr_params['mode'], min_delta=reducelr_params['min_delta'], min_lr=reducelr_params['min_lr'])
      model.fit(xtr, ytr, batch_size=bs, epochs=ep, validation_data=(xte,yte), callbacks=[stop,change_lr,checkpoint], verbose=1)
      return model
    else:
      model.fit(xtr, ytr, batch_size=bs, epochs=ep, validation_data=(xte,yte), verbose=1)
      return model



# function to plot training graphs and reduceLRonPlateau
def show_training(history, saveit=False, filename=None):
  '''
  Saves and Plots training history. Plots Loss Vs Epoch and Accuracy Vs Epoch graph as well as ReduceLrOnPlateau
  graph (if applicaple)

  history: pass model training distory. If you have previously saved training history, you can pass that or
  else pass model.history.history

  saveit: if True it saves the training history in CSV format at this path: 
  My Drive/Automatic Ticket Assignment/Model/model histories

  filename: name of the csv file to save the history with
  '''
  
  hist_df = pd.DataFrame(history)

  if(saveit==True):
    if(filename==None):
      return "Please enter a valid filename"
    else:
      hist_df.to_csv('/content/drive/My Drive/Automatic Ticket Assignment/Model/model histories/'+filename+'.csv',index=False)
      print("Training history saved!")

  loss = hist_df['loss']
  acc = hist_df['accuracy']
  val_loss = hist_df['val_loss']
  val_acc = hist_df['val_accuracy']
  epoch = hist_df.shape[0]

  figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(13, 7))

  axes[0].plot(range(0,epoch), loss, label="train loss")
  axes[0].plot(range(0,epoch), val_loss, label="test loss")
  axes[0].set_title("Loss vs Epoch")
  axes[0].set_xlabel("Epochs")
  axes[0].set_ylabel("Loss")
  axes[0].legend()

  axes[1].plot(range(0,epoch), acc, label="train acc")
  axes[1].plot(range(0,epoch), val_acc, label="test acc")
  axes[1].set_title("Accuracy vs Epoch")
  axes[1].set_xlabel("Epochs")
  axes[1].set_ylabel("Accuracy")
  axes[1].legend()

  figure.show()
  plt.tight_layout()

  if('lr' in list(hist_df.columns)):
    plt.figure(figsize=(7, 3))
    plt.plot(range(0,epoch), hist_df["lr"])
    plt.title("ReduceLROnPlateau")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.show()



# function to evaluate the metrics of the model
def DL_evaluation_metrics(ytrue, ypred_proba, multiclass, avg):
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
  print("Accuracy:",accuracy_score(ytrue.argmax(axis=1), ypred_proba.argmax(axis=1)))
  print("(macro) ROC-AUC:",roc_auc_score(y_true=ytrue, y_score=ypred_proba, multi_class=multiclass, average="macro"))
  print("("+avg+") F1 Score:",f1_score(y_true=ytrue.argmax(axis=1), y_pred=ypred_proba.argmax(axis=1), average=avg))
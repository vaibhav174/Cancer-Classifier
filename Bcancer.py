import pandas as pd
from tkinter.filedialog import *
root = Tk()
root.geometry('500x600')
root.title('Cancer Classifier')
global dfile

def fhand():
    global dfile
    dfile= askopenfile()
def des():
    root.destroy()

def main():
    df = pd.read_csv(dfile)
    num = df[['diagnosis']].groupby('diagnosis').size()
    
    from sklearn.model_selection import train_test_split
    trainingDataSet, evaluationDataSet = train_test_split(df, test_size = 0.3)

    from sklearn import svm
    classifier = svm.SVC(gamma=0.001, C=100.);
    evaluationDataSetCopy = evaluationDataSet.copy()
    trainingDataSetCopy = trainingDataSet.copy()
    
    trainingDataSetLabels = trainingDataSet[['diagnosis']];

    evaluationDataSetCopy.pop('id')
    evaluationDataSetCopy.pop('diagnosis')
    evaluationDataSetCopy.pop('Unnamed: 32')

    trainingDataSetCopy.pop('id')
    trainingDataSetCopy.pop('diagnosis')
    trainingDataSetCopy.pop('Unnamed: 32')

    trainingDataSetUnlabeled = trainingDataSetCopy;
    evaluationDataSetUnlabeled = evaluationDataSetCopy;
    evaluation = {
        'data': evaluationDataSet.as_matrix(),
        'target': evaluationDataSet.as_matrix(columns=['diagnosis']).flatten(),
        'unlabeled': evaluationDataSetUnlabeled.as_matrix(),
        }
    training = {
        'data': trainingDataSet.as_matrix(),
        'target': trainingDataSetLabels.as_matrix(columns=['diagnosis']).flatten(),
        'unlabeled': trainingDataSetUnlabeled.as_matrix(),
        }
    classifier.fit(training['unlabeled'], training['target'])
    prediction = classifier.predict(evaluation['unlabeled'][0].reshape(1, -1))[0];
    actualValue = evaluation['target'][0]

    mlabel= Label(text='Predicted Type:',font=('arial',10, 'bold')).pack(anchor=W)
    mylabel1= Label(text=prediction, font=('arial',8, 'bold')).pack(anchor=W)
    mlabel= Label(text='Actual Type:',font=('arial',10, 'bold')).pack(anchor=W)
    mylabel2= Label(text=actualValue, font=('arial',8, 'bold')).pack(anchor=W)

    if(prediction==actualValue):
        mylabel3= Label(text='Correct Prediction', font=('arial',10, 'bold')).pack()
    else:
        mylabel4= Label(text="I'm still not smart enough", font=('arial',10, 'bold')).pack()

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    predictions = classifier.predict(evaluation['unlabeled'])
    actualValues = evaluation['target']

    accuracyScore= accuracy_score(actualValues, predictions);
    classification_report = classification_report(actualValues, predictions);
    acs= round(accuracyScore*100)
    cm=confusion_matrix(actualValues, predictions)

    Label().pack()
    mylabe2= Label(text='Accuracy %:', font=('arial',12, 'bold')).pack(anchor=W)
    mylabel5= Label(text=acs, font=('arial',10)).pack(anchor=W)
    mylabel6= Label(text='Classification Report =>', font=('arial',12,'bold')).pack(anchor=W)
    mylabel7= Label(text=classification_report, font=('arial',10)).pack(anchor=W)
    mylabel8= Label(text='Confusion Matrix :', font=('arial',12,'bold')).pack(anchor=W)
    mylabel9= Label(text=cm, font=('arial',10)).pack(anchor=W)

      
mylabel= Label(text='AI Project', font=('arial',20, 'bold')).pack()
btn = Button(text='Choose Dataset', width=60, command=fhand).pack()
btn2= Button(text='Next',width=60,command=main).pack()
btn2= Button(text='Close',width=60,command=des).pack()
root.mainloop()

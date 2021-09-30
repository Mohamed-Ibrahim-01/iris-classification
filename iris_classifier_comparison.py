import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html, display

from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

def explore():
    sns.set_theme(style="ticks")
    df = sns.load_dataset("iris")
    sns.pairplot(df, hue="species")

def compare(classifiers, data):
    train, test = data
    fitted_clfs, train_reports, test_reports = [], [], []
    for clf in classifiers: 
        fitted_clfs.append(fit_data(clf, train))
    for clf in fitted_clfs: 
        train_reports.append(clf_report(clf, train))
        test_reports.append(clf_report(clf, test))
    log_reports(zip(train_reports, test_reports))
    confusion_compare(fitted_clfs, test)

def fit_data(clf, data):
    inputs, targets = data
    classifier = GridSearchCV(clf['method'](), clf['parameters'], cv=5)
    classifier.fit(inputs, targets);
    return (clf['name'], classifier)

def clf_report(clf, data):
    clf_name, classifier = clf
    inputs, lables = data
    predictions = classifier.predict(inputs)
    report = classification_report(lables, predictions, output_dict=True)
    return (clf_name, report)

def log_reports(reports):
    for report in reports:
        (clf_name, train_report),(_,test_report) = report
        display_html(f'<h3>{clf_name}</h3>', raw=True)
        train_report_df = pd.DataFrame.from_dict(train_report)
        test_report_df = pd.DataFrame.from_dict(test_report)
        display_report([train_report_df, test_report_df], ["Trian","Test"])

def display_report(dfs, names=[]):
    for i in range(0, len(names)):
        display_html(f'<h4 style="color:DodgerBlue;">{names[i]}</h4>', raw=True)
        display(dfs[i])

def confusion_compare(fitted_clfs, data):
    classifiers = [clf[1] for clf in fitted_clfs]
    inputs, targets = data
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8))

    for cls, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(cls, 
                              inputs, 
                              targets, 
                              ax=ax, 
                              cmap='Blues',
                              )
        ax.title.set_text(type(cls.estimator).__name__)
    plt.tight_layout()  
    plt.show()

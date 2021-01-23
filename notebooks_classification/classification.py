from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def compute_classifier(x_train, y_train, model):
    # create kfold
    kfold = KFold(n_splits=10, random_state=18, shuffle=True)

    # Predict y with cross validation
    y_pred = cross_val_predict(model, x_train, y_train, cv=kfold)

    # Compute metrics
    accuracy = accuracy_score(y_train, y_pred)
    print("Accuracy : " + str(accuracy))
    return y_pred

def make_classification_report(X, y, class_names, clf):
    kfold = KFold(n_splits=10, random_state=5)
    y_pred = cross_val_predict(clf, X, y, cv=kfold)

    cf_matrix = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(15,15))

    sns.heatmap(cf_matrix, annot=True, ax=ax, cmap='hot_r')
    plt.show()
    return classification_report(y, y_pred, target_names=class_names)

def plot_scores_vs_depth(depth_list, score_list, title):
    plt.plot(depth_list, score_list, label='F1-Score vs depth')
    plt.xlabel('Depth')
    plt.suptitle(title)
    plt.grid()
    plt.show()



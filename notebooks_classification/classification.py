from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def compute_classifier(x_train, y_train, model, target_names):
    # create kfold
    kfold = KFold(n_splits=10, random_state=18, shuffle=True)

    # Predict y with cross validation
    y_pred = cross_val_predict(model, x_train, y_train, cv=kfold)

    # Compute metrics
    accuracy = accuracy_score(y_train, y_pred)
    print("Accuracy : " + str(accuracy)+"\n")
    print(classification_report(y_train, y_pred, target_names=target_names))

    return y_pred

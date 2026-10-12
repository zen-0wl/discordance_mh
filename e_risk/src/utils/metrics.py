import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------
# compute metrics
# ---------------------------------------------------

def compute_metrics(

    y_true,

    y_pred,

):

    results = {

        "accuracy": accuracy_score(

            y_true,
            y_pred,

        ),

        "precision": precision_score(

            y_true,
            y_pred,

            zero_division=0,

        ),

        "recall": recall_score(

            y_true,
            y_pred,

            zero_division=0,

        ),

        "f1": f1_score(

            y_true,
            y_pred,

            zero_division=0,

        ),

        "macro_f1": f1_score(

            y_true,
            y_pred,

            average="macro",

            zero_division=0,

        ),

    }

    return results


# ---------------------------------------------------
# confusion matrix
# ---------------------------------------------------

def get_confusion_matrix(

    y_true,

    y_pred,

):

    return confusion_matrix(

        y_true,

        y_pred,

    )


# ---------------------------------------------------
# classification report
# ---------------------------------------------------

def get_classification_report(

    y_true,

    y_pred,

):

    return classification_report(

        y_true,

        y_pred,

        digits=4,

        zero_division=0,

    )


# ---------------------------------------------------
# pretty print
# ---------------------------------------------------

def print_metrics(

    metrics,

):

    print()

    print("=" * 60)

    print("Evaluation Metrics")

    print("=" * 60)

    print()

    print(f"Accuracy : {metrics['accuracy']:.4f}")

    print(f"Precision: {metrics['precision']:.4f}")

    print(f"Recall   : {metrics['recall']:.4f}")

    print(f"F1       : {metrics['f1']:.4f}")

    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")

    print()


# ---------------------------------------------------
# prediction helper
# ---------------------------------------------------

def predict_labels(

    logits,

):

    return np.argmax(

        logits,

        axis=1,

    )
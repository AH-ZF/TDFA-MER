"""
The main purpose is to analyze the results of the model training: loss, F1-score, Acc, etc.
"""

import matplotlib.pyplot as plt
import os, datetime, math
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from matplotlib import rcParams
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# GEN means generate


# Calculate the confusion matrix
def confusionMatrix(gt, pred, show=False):
    TP = 0
    for i in range(len(gt)):
        if gt[i] == pred[i] and gt[i] == 1:
            TP += 1
    FP = sum(pred) - TP
    FN = sum(gt) - TP

    if TP == 0:
        f1_score = 0
        pass
    else:
        f1_score = (2 * TP) / (2 * TP + FP + FN)
        pass
    num_samples = len([x for x in gt if x == 1])
    if num_samples != 0:
        recall = TP / num_samples
    else:
        recall = 0
        pass
    return f1_score, recall

# Other formulae principles
def paper_metric(y_true, y_pred):
    lenghtC = len(set(y_true))
    TP = []
    FP = []
    FN = []
    TN = []
    Fi = []
    Ris = []
    PiS = []
    for i in range(lenghtC):
        sums = 0
        sumstn = 0
        for j in range(len(y_pred)):
            if i == y_true[j] and i == y_pred[j]:
                sums += 1
                pass
            if i != y_true[j] and i != y_pred[j]:
                sumstn += 1
                pass
            pass
        TP.append(sums)
        FP.append(y_pred.count(i) - sums)
        FN.append(y_true.count(i) - sums)
        TN.append(sumstn)
        Pis = sums / (y_pred.count(i))
        Ri = sums / (y_true.count(i))
        Ris.append(Ri)
        PiS.append(Pis)
        Fi.append(2 * Pis * Ri / (Pis + Ri))
        pass
    allR = sum(Ris) / len(Ris)
    allP = sum(PiS) / len(PiS)
    uf1 = sum(Fi) / len(Fi)
    uar = sum(Ris) / len(Ris)
    accs=sum(TP) /(sum(TP) + sum(FP))

    if allR == 0 and allP == 0:
        print('Error：Calculation of F1 for multiple classifications shows an error of 0 in the denominator!')
        pass
    else:
        f1_score = 2 * allR * allP / (allR + allP)
        pass
    return accs,f1_score, uf1, uar



def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    for emotion, emotion_index in label_dict.items():
        gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
        pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
        try:
            f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
            f1_list.append(f1_recog)
            ar_list.append(ar_recog)
        except Exception as e:
            pass
    UF1 = np.mean(f1_list)
    UAR = np.mean(ar_list)
    return UF1, UAR



# Calculation of training phase indicators
def GENxycurve(x, y, savepath, index, imgnname):
    Cueve = os.path.join(savepath,
                         imgnname + "(" + str(index) + ").png")

    plt.figure("Curve")
    plt.title(imgnname)
    plt.plot(x, y, "r-x", markersize=5, lw=1, label="Epoch " + imgnname)

    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Epoch_" + imgnname, fontsize=10)
    plt.grid(c='k', ls='-')
    plt.legend()
    # And label each point with its coordinate value #
    for i in range(len(x) - 10, len(x), 2):
        plt.text(i, y[i], "({},{:.3f})".format(i, y[i]))
        pass
    locy = 0.5
    for i in range(len(x) - 5, len(x)):
        plt.text(len(x) - 5, locy, "({},{:.3f})".format(i, y[i]), c='r')
        locy += 0.02
        pass
    # Save the image to a local path
    plt.savefig(Cueve)
    plt.close()


    pass

# Plot the confusion matrix:
def GENconfusion_matrix(y_true, y_pred, savepath, index, imgnname):
    config = {
        "font.family": 'Times New Roman',
        "font.size": 10,
        "mathtext.fontset": 'stix',
        "font.weight": 'bold',
        "figure.dpi": 600
    }

    configxylabel = {
        "family": 'Times New Roman',
        "size": 10,
        "weight": 'bold'
    }
    rcParams.update(config)
    confMfig = os.path.join(savepath,
                            imgnname + "(" + str(index) + ")" + ".png")

    configpng = os.path.join(savepath,
                             imgnname + "(" + str(index) + ")" + 'Normalize' + ".png")
    configtif = os.path.join(savepath,
                             imgnname + "(" + str(index) + ")" + 'Normalize' + ".tif")

    classes = ["Negative", "Positive", "Surprise"]

    # Calculate the output confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(include_values=True, cmap=plt.cm.Reds, colorbar=True)
    # Save the confusion matrix image
    plt.savefig(confMfig)
    plt.close()
    # Normalize the computed confusion matrix
    np.set_printoptions(precision=4, suppress=True)
    cmN = np.zeros((cm.shape[0], cm.shape[1]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cmN[i][j] = cm[i][j] / sum(cm[i])
            pass
        pass
    dispN = ConfusionMatrixDisplay(confusion_matrix=cmN, display_labels=classes)
    dispN.plot(include_values=True, cmap=plt.cm.Reds, colorbar=True,values_format='.4f')
    # Save the confusion matrix image
    plt.xlabel('Predict label', fontdict=configxylabel)
    plt.ylabel('True label', fontdict=configxylabel)
    plt.savefig(configpng, dpi=600)
    plt.savefig(configtif, dpi=600)
    plt.close()
    pass




# Calculate Acc:
def GENaccuracy_score(y_true, y_pred):
    Acc = accuracy_score(y_true, y_pred, normalize=True)
    return Acc


# Calculate the F1-score:
def GENf1_score(y_true, y_pred):
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return micro_f1


# Manual calculation of f1 for multiple classifications
def self_f1(y_true, y_pred):
    lenghtC = len(set(y_true))
    TP = []
    FP = []
    FN = []
    TN = []
    for i in range(lenghtC):
        sums = 0
        sumstn = 0
        for j in range(len(y_pred)):
            if i == y_true[j] and i == y_pred[j]:
                sums += 1
                pass
            if i != y_true[j] and i != y_pred[j]:
                sumstn += 1
                pass
            pass
        TP.append(sums)
        FP.append(y_pred.count(i) - sums)
        FN.append(y_true.count(i) - sums)
        TN.append(sumstn)
        pass
    allR = sum(TP) / (sum(TP) + sum(FN))
    allP = sum(TP) / (sum(TP) + sum(FP))
    if allR == 0 and allP == 0:
        print('Error：Calculation of F1 for multiple classifications shows an error of 0 in the denominator!')
        pass
    else:
        f1_score = 2 * allR * allP / (allR + allP)
        pass
    return f1_score



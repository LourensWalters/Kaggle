import matplotlib.pyplot as plt
import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import auc
from numpy import set_printoptions

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Creates a plot for the specified confusion matrix object and calculates relevant accuracy measures.
    """

    # Add Normalization option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=18)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

    fp_label = 'false positive'
    fp = cm[0][1]
    fn_label = 'false negative'
    fn = cm[1][0]
    tp_label = 'true positive'
    tp = cm[1][1]
    tn_label = 'true negative'
    tn = cm[0][0]

    tpr_label = 'sensitivity'
    tpr = round(tp / (tp + fn), 3)
    tnr_label = 'specificity'
    tnr = round(tn / (tn + fp), 3)
    ppv_label = 'precision'
    ppv = round(tp / (tp + fp), 3)
    npv_label = 'npv'
    npv = round(tn / (tn + fn), 3)
    fpr_label = 'fpr'
    fpr = round(fp / (fp + tn), 3)
    fnr_label = 'fnr'
    fnr = round(fn / (tp + fn), 3)
    fdr_label = 'fdr'
    fdr = round(fp / (tp + fp), 3)

    acc_score = round((tp + tn) / (tp + fp + tn + fn), 3)

    print('\naccuracy:\t\t\t{}  \nprecision:\t\t\t{} \nsensitivity:\t\t\t{}'.format(acc_score, ppv, tpr))
    print('\nspecificity:\t\t\t{} \nnegative predictive value:\t{}'.format(tnr, npv))
    print('\nfalse positive rate:\t\t{}  \nfalse negative rate:\t\t{} \nfalse discovery rate:\t\t{}'.format(fpr, fnr,
                                                                                                            fdr))
def plot_roc_curve(fpr, tpr, title="Receiver operating characteristic (ROC) Curve"):
    """
    Creates a plot for the specified roc curve object.
    """

    # Visualization for ROC curve
    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10,8))
    lw = 2
    _ = plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve');
    _ = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--');
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    _ = plt.show();

def plot_feature_importance_log(fit, features):
    """
    Creates a plot for the specified feature importance object.
    """

    set_printoptions(precision=3)

    # Summarize selected features
    scores = -np.log10(fit.pvalues)
    #scores /= scores.max()

    importances = np.array(scores)
    feature_list = features
    sorted_ID=np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

    for i,v in enumerate(reverse_importances):
        print('Feature: %20s\tScore:\t%.5f' % (reverse_features[i],v))

    # Plot feature importance
    #sns.set(font_scale=1);
    _ = plt.figure(figsize=[10,10]);
    _ = plt.xticks(rotation='horizontal', fontsize=20)
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.yticks(fontsize=20)
    #plt.tight_layout()
    _ = plt.show();

def plot_feature_importance_dec(fit, features):
    """
    Creates a plot for the specified feature importance object.
    """

    set_printoptions(precision=3)

    # Summarize selected features
    scores = fit
    #scores /= scores.max()

    importances = np.array(scores)
    feature_list = features
    sorted_ID=np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

    for i,v in enumerate(reverse_importances):
        print('Feature: %20s\tScore:\t%.5f' % (reverse_features[i],v))

    # Plot feature importance
    #sorted_ID=np.array(np.argsort(scores)[::-1])
    #sns.set(font_scale=1);
    _ = plt.figure(figsize=[10,10]);
    _ = plt.xticks(rotation='horizontal', fontsize=20)
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.yticks(fontsize=20)
    _ = plt.show();

    #_=plt.bar(X_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
    #plt.show()

def plot_feature_importance(fit, features):
    """
    Creates a plot for the specified feature importance object.
    """

    set_printoptions(precision=3)

    # Summarize selected features
    scores = -np.log10(fit.pvalues_)

    importances = np.array(scores)
    feature_list = features
    sorted_ID=np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

    for i,v in enumerate(reverse_importances):
        print('Feature: %20s\tScore:\t%.5f' % (reverse_features[i],v))

    # Plot feature importance
    #sns.set(font_scale=1);
    _ = plt.figure(figsize=[10,10]);
    _ = plt.xticks(rotation='horizontal', fontsize=20)
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.yticks(fontsize=20)
    _ = plt.show();

def plotAge(df, axes, single_plot=True):

    if (single_plot):
        sns.kdeplot (data=df.loc[(df['ca_disease'] == 0), 'age'], shade = True, label = 'Disease False')
        sns.kdeplot (data=df.loc[(df['ca_disease'] == 1), 'age'], shade = True, label = 'Disease True')
        plt.xlabel('Age', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)
        plt.show()

    else:
        facet_grid = sns.FacetGrid(df, hue='ca_disease')
        _ = facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0]);

        legend_labels = ['disease false', 'disease true']
        for t, l in zip(axes[0].get_legend().texts, legend_labels):
            t.set_text(l)
            axes[0].set(xlabel='age', ylabel='density')

        avg = df[["age", "ca_disease"]].groupby(['age'], as_index=False).mean();
        _ = sns.barplot(x='age', y='ca_disease', data=avg, ax=axes[1]);
        _ = axes[1].set(xlabel='age', ylabel='disease probability');

    #plt.clf()

def plotCategorical(attribute, labels, ax_index, df, axes):
    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='ca_disease', hue=attribute, data=df, ax=axes[ax_index][1])
    avg = df[[attribute, 'ca_disease']].groupby([attribute], as_index=False).mean()
    _ = sns.barplot(x=attribute, y='ca_disease', hue=attribute, data=avg, ax=axes[ax_index][2])

    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
        t.set_text(l)
    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
        t.set_text(l)

def plotContinuous(attribute, xlabel, ax_index, df, axes):
    if (ax_index == 5):
        return
    _ = sns.distplot(df[[attribute]], ax=axes[ax_index][0]);
    _ = axes[ax_index][0].set(xlabel=xlabel, ylabel='density');
    axes[ax_index][0].xaxis.label.set_size(24)
    axes[ax_index][0].yaxis.label.set_size(24)
    axes[ax_index][0].tick_params('y', labelsize = 20);
    axes[ax_index][0].tick_params('x', labelsize = 20);
    _ = sns.violinplot(x='ca_disease', y=attribute, data=df, ax=axes[ax_index][1]);
    axes[ax_index][1].xaxis.label.set_size(24)
    axes[ax_index][1].yaxis.label.set_size(24)
    axes[ax_index][1].tick_params('y', labelsize = 20);
    axes[ax_index][1].tick_params('x', labelsize = 20);
    plt.tight_layout()

def plotVar(isCategorical, categorical, continuous, df, axes):
    if isCategorical:
        [plotCategorical(x[0], x[1], i, df, axes) for i, x in enumerate(categorical)]
    else:
        [plotContinuous(x[0], x[1], i, df, axes) for i, x in enumerate(continuous)]

def main():
    from sklearn.metrics import confusion_matrix
    """
    main function - does all the work
    """
    # parse arguments
    cnf_matrix = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])

    # generate plots
    plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)

if __name__ == "__main__":
    # call main
    main()
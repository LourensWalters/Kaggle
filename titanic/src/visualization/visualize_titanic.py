import matplotlib.pyplot as plt
import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import auc
from numpy import set_printoptions
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5), learn_scoring=None, scoring_title="Score"):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=learn_scoring)

    if learn_scoring == "neg_mean_squared_error":
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
    else:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_ylabel(scoring_title)

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel(scoring_title)
    axes[2].set_title("Performance of the model")

    return plt

def get_validation_curve(classifier, X_data, y_data, param_feed, cv_splits=10,
                         figure_size=(7, 5), x_scale='linear', y_lim=[0.70, 1.0]):
    """
    Generates a validation curve over a specified range of hyperparameter values for a given classifier,
    and prints the optimal values yielding i) the highest cross-validated mean accuracy and ii) the smallest
    absolute difference between the mean test and train accuracies (to assess overfitting).

    : param classifier : Classifier object, assumed to have a scikit-learn wrapper.

    : param X_data : Pandas dataframe containing the training feature data.

    : param y_data : Pandas dataframe containing the training class labels.

    : param param_feed : Dictionary of form {'parameter_name' : parameter_values}, where parameter_values
                         is a list or numpy 1D-array of parameter values to sweep over.

    : param cv_splits : Integer number of cross-validation splits.

    : param figure_size : Tuple of form (width, height) specifying figure size.

    : param x_scale : String, 'linear' or 'log', controls x-axis scale type of plot.

    : param y_lim : List of form [y_min, y_max] for setting the plot y-axis limits.

    : return : None.

    """
    base_param_name = list(param_feed.keys())[0]
    param_range_ = param_feed[base_param_name]

    piped_clf = Pipeline([('clf', classifier)]) # I use this merely to assign the handle 'clf' to our classifier

    # Obtain the cross-validated scores as a function of hyperparameter value
    train_scores, test_scores = validation_curve(estimator=piped_clf,
                                                 X=X_data,
                                                 y=y_data,
                                                 param_name='clf__' + base_param_name,
                                                 param_range=param_range_,
                                                 cv=cv_splits)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Generate the validation curve plot
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figure_size)

    plt.plot(param_range_, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(param_range_, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range_, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(param_range_, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.xscale(x_scale)
    plt.xlabel(base_param_name)
    plt.ylim(y_lim)
    plt.ylabel('Accuracy')
    plt.grid(b=True, which='major', color='black', linestyle=':')
    plt.legend(loc='lower right')
    plt.title('Validation Curve for Parameter ' + base_param_name)
    plt.show()

    # Display optimal parameter values for best accuracy and smallest train-test difference
    diffs = abs(train_mean - test_mean)
    id_best_diff = np.argmin(diffs)
    id_best_acc = np.argmax(test_mean)

    print('Best Accuracy is %.5f occuring at %s = %s' % (test_mean[id_best_acc],
                                                         base_param_name,
                                                         param_range_[id_best_acc]))


    print('Smallest Train-Test Difference is %.5f occuring at %s = %s' % (diffs[id_best_diff],
                                                                          base_param_name,
                                                                          param_range_[id_best_diff]))

    return

def get_learning_curve(classifier, X_data, y_data, training_sizes=np.linspace(0.1, 1.0, 10), cv_splits=10,
                       figure_size=(7, 5), y_lim=[0.70, 1.0]):
    """
    Generates a learning curve to asses bias-variance tradeoff by plotting cross-validated train and test
    accuracies as a function of the number of samples used for training.

    : param classifier : Classifier object, assumed to have a scikit-learn wrapper.

    : param X_data : Pandas dataframe containing the training feature data.

    : param y_data : Pandas dataframe containing the training class labels.

    : param training_sizes : Numpy 1D array of the training sizes to sweep over, specified as fractions
                             of the total training set size.

    : param cv_splits : Integer number of cross-validation splits.

    : param figure_size : Tuple of form (width, height) specifying figure size.

    : param y_lim : List of form [y_min, y_max] for setting the plot y-axis limits.

    : return : None.

    """
    # Obtain the cross-validated scores as a function of the training size
    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=X_data,
                                                            y=y_data,
                                                            train_sizes=training_sizes,
                                                            cv=cv_splits,
                                                            n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Generate the learning curve plot
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figure_size)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='red', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='red')

    plt.xlabel('Number of Training Samples')
    plt.ylim(y_lim)
    plt.ylabel('Accuracy')
    plt.grid(b=True, which='major', color='black', linestyle=':')
    plt.legend(loc=4)
    plt.title('Learning Curve')
    plt.show()

    return

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
        sns.kdeplot (data=df.loc[(df['survived'] == 0), 'age'], shade = True, label = 'Died')
        sns.kdeplot (data=df.loc[(df['survived'] == 1), 'age'], shade = True, label = 'Survived')
        plt.xlabel('Age', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)
        plt.show()

    else:
        facet_grid = sns.FacetGrid(df, hue='survived')
        _ = facet_grid.map(sns.kdeplot, "age", shade=True, ax=axes[0]);

        legend_labels = ['died', 'survived']
        for t, l in zip(axes[0].get_legend().texts, legend_labels):
            t.set_text(l)
            axes[0].set(xlabel='age', ylabel='density')

        avg = df[["age", "survived"]].groupby(['age'], as_index=False).mean();
        _ = sns.barplot(x='age', y='survived', data=avg, ax=axes[1]);
        _ = axes[1].set(xlabel='age', ylabel='survival probability');

    #plt.clf()

def plotCategorical(attribute, labels, ax_index, df, axes):
    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='survived', hue=attribute, data=df, ax=axes[ax_index][1])
    avg = df[[attribute, 'survived']].groupby([attribute], as_index=False).mean()
    _ = sns.barplot(x=attribute, y='survived', hue=attribute, data=avg, ax=axes[ax_index][2])

    for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
        t.set_text(l)
    for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
        t.set_text(l)

def plotContinuous(attribute, xlabel, ax_index, df, axes):
    if (ax_index == 5):
        return
#    _ = sns.distplot(df[[attribute]], ax=axes[ax_index][0]);
    g = sns.displot(x=attribute, data=df[[attribute]]);
#   g.axes_dict = axes
    _ = axes[ax_index][0].set(xlabel=xlabel, ylabel='density');
    axes[ax_index][0].xaxis.label.set_size(24)
    axes[ax_index][0].yaxis.label.set_size(24)
    axes[ax_index][0].tick_params('y', labelsize = 20);
    axes[ax_index][0].tick_params('x', labelsize = 20);
#    _ = sns.violinplot(x='survived', y=attribute, data=df, ax=axes[ax_index][1]);
    h = sns.violinplot(x='survived', y=attribute, data=df);
#    h.axes_dict = axes[ax_index][1]
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
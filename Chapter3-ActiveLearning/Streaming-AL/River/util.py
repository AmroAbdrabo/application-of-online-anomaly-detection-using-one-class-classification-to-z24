"""
Contains utilities to plot and/or compute ROC, AUC, F1, accuracy
"""

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def all_auc(xrange, m_all_exp):
    iter_all_exp = []
    for exp in m_all_exp:
        # exp is a dictionary
        iter_one_exp = []
        for sample_size in xrange:
            iter_one_exp.append(roc_auc_score(exp[sample_size][0], exp[sample_size][2]))
        iter_all_exp.append(iter_one_exp)
    return iter_all_exp

def all_f1_score(xrange, m_all_exp):
    iter_all_exp = []
    for exp in m_all_exp:
        # exp is a dictionary
        iter_one_exp = []
        for sample_size in xrange:
            iter_one_exp.append(f1_score(exp[sample_size][0], exp[sample_size][1]))
        iter_all_exp.append(iter_one_exp)
    return iter_all_exp

def all_accu_score(xrange, m_all_exp):
    iter_all_exp = []
    for exp in m_all_exp:
        # exp is a dictionary
        iter_one_exp = []
        for sample_size in xrange:
            iter_one_exp.append(balanced_accuracy_score(exp[sample_size][0], exp[sample_size][1]))
        iter_all_exp.append(iter_one_exp)
    return iter_all_exp

def all_precision_score(xrange, m_all_exp):
    iter_all_exp = []
    for exp in m_all_exp:
        # exp is a dictionary
        iter_one_exp = []
        for sample_size in xrange:
            iter_one_exp.append(precision_score(exp[sample_size][0], exp[sample_size][1]))
        iter_all_exp.append(iter_one_exp)
    return iter_all_exp

def all_recall_score(xrange, m_all_exp):
    iter_all_exp = []
    for exp in m_all_exp:
        # exp is a dictionary
        iter_one_exp = []
        for sample_size in xrange:
            iter_one_exp.append(recall_score(exp[sample_size][0], exp[sample_size][1]))
        iter_all_exp.append(iter_one_exp)
    return iter_all_exp

def fill_between_plot(x_range, average_obs, std_obs, label, title, xlabel):
    plt.figure(figsize=(8, 5))
    colors = ['blue', 'cyan', 'red', 'green']
    for idx, (avg_obs, std_ob) in enumerate(zip(average_obs, std_obs)):
        plt.plot(x_range, avg_obs, color = colors[idx], label=label)
        plt.fill_between(x_range, avg_obs - std_ob, avg_obs + std_ob, color=colors[idx], alpha=0.5, label='1 std dev')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(label)
    plt.legend()
    plt.show()

def fill_between_plot_diffx(x_ranges, average_obs, std_obs, label, title, labely, xlabel):
    plt.figure(figsize=(8, 5))
    colors = ['blue', 'red', 'cyan', 'green']
    
    # Ensure that x_ranges, average_obs, and std_obs have the same length
    if not (len(x_ranges) == len(average_obs) == len(std_obs)):
        raise ValueError("Length of x_ranges, average_obs, and std_obs must be the same")

    for idx, (x_range, avg_obs, std_ob) in enumerate(zip(x_ranges, average_obs, std_obs)):
        plt.plot(x_range, avg_obs, color=colors[idx % len(colors)], label=f'{label[idx]}')
        plt.fill_between(x_range, avg_obs - std_ob, avg_obs + std_ob, color=colors[idx % len(colors)], alpha=0.5, label='1 std dev')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(labely)
    plt.legend()
    plt.show()
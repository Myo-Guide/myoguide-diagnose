import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.cm import ScalarMappable
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

"""This module provides the core functionality for myoguide-diagnose analysis.

It includes functions to process data, analyze patterns, and generate diagnostics.
"""

def _diverging_score_cmap():
    colors = [
        (13/255, 232/255, 255/255),
        (72/255, 75/255, 170/255),
        (89/255, 89/255, 89/255),
        (150/255, 28/255, 34/255),
        (255/255, 12/255, 155/255),
    ]
    cmap_name = 'cool_warm_light'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=10000)
    return cmap

def counts_plot(df:pd.DataFrame, target:str, save_dir:str=None, save_name:str=None) -> None:
    '''Shows a barplot of the different classes in the data 

    Parameters
    ----------
    df
        Input dataframe
    target
        Column containing the class labels
    save_dir, optional
        Directory to save the plot. If None, the plot is not saved. By default None
    save_name, optional
        _description_, by default None
    '''
    _df = df.copy()
    fig, ax = plt.subplots(figsize=(16, 3))
    large_to_small = _df.groupby(target).size().sort_values().index[::-1]
    sns.countplot(x=target, data=_df, order=large_to_small)
    plt.title('order from largest to smallest')
    ax.tick_params(axis='x', rotation=45)
    ax.bar_label(ax.containers[0])
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{save_name}.png')

    plt.show()

def heatmap_legacy(
    df, 
    id_col, 
    cols, 
    target_col,
    mode:str='absolute',
    cluster:bool=False,
    save_dir:str=None,
    save_name:str=None,
    calc_mean:bool=True
):
    if mode not in ['absolute', 'mean']:
        raise Exception('Mode must be `absolute` or `mean`')

    _df = df.copy()
    _cols = cols.copy()

    _df[id_col] = _df[id_col].astype('category')

    mercuri_df = _df[_cols]

    if calc_mean:
        _df['mean'] = mercuri_df.mean(axis=1)
        
    _df.sort_values(by=["mean"], axis=0, inplace=True, ascending=False)

    _df.reset_index(drop=True, inplace=True)

    for d in list(_df[target_col].unique()):
        plt.figure(figsize=(10, 10))
        if mode == 'absolute' and not cluster:
            sns.heatmap(data=_df.loc[_df[target_col] == d][_cols], vmin=0, vmax=100)
        if mode == 'absolute' and cluster:
            # Does not work if there are nans
            sns.clustermap(data=_df.loc[_df[target_col] == d][_cols], col_cluster=False)
        elif mode == 'mean' and not cluster:
            sns.heatmap(data=_df.loc[_df[target_col] == d][_cols], cmap="coolwarm", vmin=-100, vmax=100, center=0)
        plt.title(d)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/{save_name}_{d}.png')

        plt.show()

def heatmap_clustered(df, config, target, t):
    '''Experimental
    '''
    from scipy.cluster.hierarchy import fcluster
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt

    cmap = _diverging_score_cmap()

    # Custom NaN-resistant Euclidean Distance Function
    def nan_euclidean(u, v):
        nan_mask = ~np.isnan(u) & ~np.isnan(v)
        u_nonan = u[nan_mask]
        v_nonan = v[nan_mask]
        return np.sqrt(np.sum((u_nonan - v_nonan) ** 2))

    _df = df.loc[df[config['label_col']] == target]
    _df.reset_index(drop=True, inplace=True)
    _mean = _df['mean'].copy()
    _df = _df[config['_muscle_columns_processed']].copy()

    # Compute the distance matrix, using the custom NaN-resistant Euclidean distance
    distance_matrix = pdist(_df.values, lambda u, v: nan_euclidean(u, v))

    # Perform hierarchical clustering
    Z = linkage(squareform(distance_matrix), 'ward')

    # Decide the number of clusters, for example by using the max distance or setting a threshold
    # Here, t is the threshold for the clustering, you might adjust it based on your dendrogram or specific needs
    # Using 'maxclust' or 'distance' criterion can change the results, adjust based on your needs
    clusters = fcluster(Z, t, criterion='maxclust')

    # Step 3 & 4: Sort clusters by their average value, and sort samples within clusters
    # Create a DataFrame that includes the cluster assignment and the means
    cluster_df = pd.DataFrame({'Cluster': clusters, 'Mean': _mean, 'Index': np.arange(len(_mean))})
    sorted_cluster_df = cluster_df.sort_values(['Cluster', 'Mean'], ascending=[True, False])

    # Use the sorted indices to sort the original DataFrame
    sorted_df = _df.iloc[sorted_cluster_df['Index'].values]

    # Generate the heatmap
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 3, 1]})

    # Plot Dendrogram
    dendrogram(Z, ax=axes[0], orientation='left', labels=sorted_df.index, leaf_font_size=12)
    axes[0].invert_yaxis()  # Invert the y-axis to match the order of the heatmap

    # Plot Heatmap
    sns.heatmap(sorted_df, ax=axes[1], cmap=cmap, vmin=-100, vmax=100, center=0, cbar_kws={'label': 'Intensity'})
    axes[1].set_title('Heatmap of Cluster-Sorted DataFrame')
    axes[1].set_yticks([])  # Hide the y-ticks as they're already shown in the dendrogram

    # Plot Mean values
    # Sort the 'mean' array according to the sorted DataFrame index
    sorted_means = _mean[sorted_df.index] + 100
    axes[2].barh(range(len(sorted_means)), sorted_means, color='skyblue')
    axes[2].set_yticks([])  # Hide the y-ticks as they're already shown in the dendrogram
    axes[2].invert_yaxis()  # Ensure the order matches the dendrogram and heatmap
    axes[2].set_title('Mean Values')

    # Adjust layout
    plt.tight_layout()

    plt.show()

def heatmap(
    df:pd.DataFrame,
    target:str,
    config:dict,
    figsize=None,
    save:bool=False,
    show_asymmetry:bool=True,
    show_age:bool=True,
    show_sex:bool=True,
    font_size:int=14,
    show_y_labels=False,
    subset_by:tuple=None,
    age_vrange:tuple=None,
    score_range=(-100, 100),
):
    '''Plots a heatmap of the data.

    .. warning:: To work properly, the configuration file must set ``scale_mean=True`` and ``scale_min_max: False``.

    Parameters
    ----------
    df
        Input dataframe loaded with :func:`mgdiagnose.process.process.read_csv`
    target
        Target class to show
    config
        Configuration dict
    figsize, optional
        Sets the vertical size of the figure based on the amout of samples, by default None
    save, optional
        Save the figure, by default False
    show_y_lables, optional
        If False, no y labels are shown
        If True, the record id is show
        If str, the column with the provided name is used as a label column
        By default False
    subset_by
        If a tuple is provided, the first element is used as a column key and the second and de expected value
    '''
    score_cmap = _diverging_score_cmap()

    if target == "show-all":
        _df = df.copy()
    else:
        _df = df.loc[df[config['label_col']] == target].copy()

    if subset_by is not None:
        _df = _df.loc[_df[subset_by[0]] == subset_by[1]].copy()
    _df.sort_values(by=['mean'], axis=0, inplace=True, ascending=False)

    if isinstance(show_y_labels, str):
        y_labels = _df[show_y_labels].tolist()
    elif show_y_labels == True:
        y_labels = _df["id"].tolist()
    else:
        y_labels = []

    scores = _df[config['_muscle_columns_processed']].copy()
    if figsize is None:
        figsize = 1.
    elif isinstance(figsize, (float, int)) :
        r = int(0.3 * (len(config['_muscle_columns_processed']) + 10))
        c = int(0.05 * scores.shape[0] * figsize)
        figsize = (r, c)
    elif isinstance(figsize, tuple) :
        pass

    plt.rcParams.update({'font.size': font_size}) 

    total_axes = 2 + show_asymmetry + show_age + show_sex
    width_ratios = []
    width_ratios.append(0.20)
    width_ratios.append(8)
    if show_asymmetry: width_ratios.append(0.4)
    if show_age: width_ratios.append(0.20)
    if show_sex: width_ratios.append(0.20)
    fig, axs = plt.subplots(1, total_axes, figsize=figsize, gridspec_kw={'width_ratios': width_ratios}, sharey=False)

    # MEAN
    ax = axs[0]
    mean = _df[['mean']].copy()
    sns.heatmap(mean, ax=ax, cmap='viridis', vmin=0, vmax=100, cbar=False)
    ax.tick_params(axis='x', rotation=90)
    if show_y_labels:
        ax.set_yticks(ticks=[i + 0.5 for i in range(len(y_labels))])
        ax.set_yticklabels(y_labels, rotation=0)
    else:
        ax.set_yticks([])

    # SCORE #
    ax = axs[1]
    sns.heatmap(scores, ax=ax, cmap=score_cmap, vmin=score_range[0], vmax=score_range[1], center=0, cbar=False)
    ax.tick_params(axis='x', rotation=90)
    ax.set_yticks([])

    # ASYMM
    if show_asymmetry:
        ax = axs[2]
        asymm = _df[['asymm_mean', 'asymm_std']].copy()
        sns.heatmap(asymm, ax=ax, cmap='plasma', vmin=0, vmax=100, cbar=False)
        ax.tick_params(axis='x', rotation=90)
        ax.set_yticks([])
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['asymm_mean', 'asymm_std'])

    # AGE #
    if show_age:
        age = _df[['age']].copy()
        ax = axs[3]
        if age_vrange is None:
            sns.heatmap(age, ax=ax, cmap='jet', cbar=False)
        else:
            sns.heatmap(age, ax=ax, cmap='jet', cbar=False, vmin=age_vrange[0], vmax=age_vrange[1])
        ax.tick_params(axis='x', rotation=90)
        ax.set_yticks([])

    # SEX #
    if show_sex:
        ax = axs[4]
        sex = _df[['patient__sex']].copy()
        sex_cmap = ListedColormap(['#ff9147', '#47b5ff'])
        sex_norm = BoundaryNorm([0, 1], sex_cmap.N, clip=True)
        sns.heatmap(sex, ax=ax, cmap=sex_cmap, norm=sex_norm, cbar=False)  
        ax.tick_params(axis='x', rotation=90)
        ax.set_yticks([])
        legend_patches = [plt.Line2D([0], [0], color=color, marker='s', markersize=10, linestyle='', label=label) for label, color in {'M': '#ff9147', 'F': '#47b5ff'}.items()]


    colorbar_axes_positions = []
    colorbar_axes_positions.append([1.0, 0.85, 0.01, 0.15])
    colorbar_axes_positions.append([1.0, 0.65, 0.01, 0.15])
    if show_asymmetry: 
        colorbar_axes_positions.append([1.0, 0.45, 0.01, 0.15])
    if show_age: 
        colorbar_axes_positions.append([1.0, 0.25, 0.01, 0.15])
    if show_sex: 
        colorbar_axes_positions.append([1.0, 0.05, 0.01, 0.15])

    cbar_labels = []
    cbar_labels.append('mean')
    cbar_labels.append('scores')
    if show_asymmetry: cbar_labels.append('asymmetry')
    if show_age: cbar_labels.append('age')

    # Create and draw colorbars
    for i, cax in enumerate(axs):
        if not i == 4 :
            cbar_ax = fig.add_axes(colorbar_axes_positions[i])
            cbar = fig.colorbar(cax.collections[0], cax=cbar_ax)
            cbar.set_label(cbar_labels[i])

    if show_sex:
        sex_norm_cbar = BoundaryNorm([0, 0.5, 1], sex_cmap.N, clip=True)
        sm = ScalarMappable(cmap=sex_cmap, norm=sex_norm_cbar)
        sm.set_array([])  # You can set an empty array as the data range
        cbar_ax = fig.add_axes(colorbar_axes_positions[-1])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1])
        cbar.set_ticklabels(['M', 'F'])
        cbar.set_label('sex')

    axs[1].set_title(f'Data overview for {target}')

    # Adjust subplots to fit colorbars
    # plt.subplots_adjust(hspace=0.1)
    plt.tight_layout()

    if save:
        plt.savefig(f'{target}.tiff', dpi=300, bbox_inches='tight')

    plt.show()   

def plot_heatmap_probs(
        df, target, 
        shap_values, muscle_cols, le, 
        sort_mode='mean', cluster_method='average', cluster_by='scores', p=10, 
        scale_h=0.1, hlines=[], 
        save=False, classifier_str=None
        ):
    
    if sort_mode not in ['mean', 'cluster']: 
        raise Exception('`sort_mode` must be `mean` or `cluster`')
    if cluster_method not in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
        raise Exception('`cluster_method` must be `single`, `complete`, `average`, `weighted`, `centroid`, `median` or `ward`')
    if cluster_by not in ['scores', 'probs', 'shap']:
        raise Exception('`cluster_by` must be `scores`, `shap`, `probs`')
    
    from scipy.spatial.distance import pdist
    from shap.plots.colors._colors import red_white_blue
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

    colors = [
        (0/255, 102/255, 235/255),
        (112/255, 112/255, 112/255),
        (235/255, 133/255, 0/255)
    ]
    asym_cmap = LinearSegmentedColormap.from_list('asym', colors, N=500)

    diagnosis_idx = le.transform([target])[0]

    _df = df.loc[df['diagnosis']==diagnosis_idx].copy()

    if sort_mode == 'cluster':
        if cluster_by == 'shap':
            _shap_values = shap_values[diagnosis_idx][_df.index]
            distance_matrix = pdist(_shap_values, metric='euclidean')
            
            Z = linkage(distance_matrix, cluster_method)
            order = leaves_list(Z)
            _df = _df.iloc[order].copy()
        else:
            if cluster_by == 'scores':
                _cluster_by = muscle_cols
            elif cluster_by == 'probs':
                _cluster_by = le.classes_

            # For columns, transpose the DataFrame: pdist(df.T.values, metric='euclidean')
            df_cluster = _df.loc[:, _df.columns.isin(_cluster_by)].copy()
            distance_matrix = pdist(df_cluster.values, metric='euclidean')
            
            Z = linkage(distance_matrix, cluster_method)
            order = leaves_list(Z)
            _df = _df.iloc[order].copy()

    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 3, 1]) 
    plt.figure(figsize=(14, len(_df) * scale_h))

    if sort_mode == 'cluster':
        ax00 = plt.subplot(gs[0,0])
        row_dendr = dendrogram(Z, orientation='left', ax=ax00, truncate_mode='level', p=p, color_threshold=np.inf, link_color_func=lambda _:'grey')

        ax10 = plt.subplot(gs[1,0])
        row_dendr = dendrogram(Z, orientation='left', ax=ax10, truncate_mode='level', p=p, color_threshold=np.inf, link_color_func=lambda _:'grey')

    ax01 = plt.subplot(gs[0,1])
    mercuri_im = ax01.imshow(_df.loc[:, _df.columns.isin(muscle_cols)], aspect='auto', interpolation='none', cmap=asym_cmap, vmin=-100, vmax=100)
    ax01.set_xticks(np.arange(len(muscle_cols)), labels=muscle_cols)
    ax01.set_yticks([], [])
    ax01.grid(c='grey')
    for hl in hlines: ax01.axhline(y=hl, color='#ffffff', linestyle=':')

    ax02 = plt.subplot(gs[0,2], sharey = ax01)
    probs_im = ax02.imshow(_df.loc[:, _df.columns.isin(le.classes_)], aspect='auto', interpolation='none', cmap='plasma')
    ax02.set_xticks(np.arange(len(le.classes_)), labels=le.classes_)
    ax02.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax02.set_yticks([], [])
    ax02.grid(c='grey')
    for hl in hlines: ax02.axhline(y=hl, color='#ffffff', linestyle=':')

    ax11 = plt.subplot(gs[1,1])
    _shap_values = shap_values[diagnosis_idx][_df.index]
    max_shap = max([abs(_shap_values.min()), abs(_shap_values.max())])
    shap_im = ax11.imshow(_shap_values, aspect='auto', interpolation='none', cmap=red_white_blue, vmin=-max_shap, vmax=max_shap)
    ax11.set_xticks(np.arange(len(muscle_cols)), labels=muscle_cols)
    ax11.set_yticks([], [])
    ax11.grid(c='grey')
    for hl in hlines: ax11.axhline(y=hl, color='#000000', linestyle=':')

    ax12 = plt.subplot(gs[1,2])
    ax12.axis('off')
    
    if sort_mode == 'cluster':
        ax00.axis('off')
        ax10.axis('off')
    ax01.get_xaxis().set_ticks([])
    plt.setp(ax11.get_xticklabels(), rotation=90, ha="right", va="center", rotation_mode="anchor")
    plt.setp(ax02.get_xticklabels(), rotation=90, ha="left", va="center", rotation_mode="anchor")

    axins_shap = inset_axes(
    ax12,
    width="5%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(0., 0., 1, 1),
    bbox_transform=ax12.transAxes,
    borderpad=0,
    )

    axins_mercuri = inset_axes(
    ax12,
    width="5%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(0.4, 0., 1, 1),
    bbox_transform=ax12.transAxes,
    borderpad=0,
    )

    axins_probs = inset_axes(
    ax12,
    width="5%",  # width: 5% of parent_bbox width
    height="100%",  # height: 50%
    loc="lower left",
    bbox_to_anchor=(0.8, 0., 1, 1),
    bbox_transform=ax12.transAxes,
    borderpad=0,
    )

    shap_cbar = plt.colorbar(shap_im, cax=axins_shap)
    shap_cbar.set_label('Shap Value')
    mercuri_cbar = plt.colorbar(mercuri_im, cax=axins_mercuri)
    mercuri_cbar.set_label('Feature Value')
    probs_cbar = plt.colorbar(probs_im, cax=axins_probs)
    probs_cbar.set_label('Diagnosis Probability')

    plt.suptitle(target)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(wspace=.02)

    if save:
        plt.savefig(f"{classifier_str}_{target}_shap_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_conf_matrix(cm, le, norm=False, save=False, classifier_str=None, figsize=(11,9)):
    _cm = cm.copy()

    if norm:
        _cm = (_cm / _cm.sum(axis=1, keepdims=True)) * 100
        _fmt = '.0f'
        _vmin, _vmax = 0, 100
    else:
        _fmt = 'g'
        _vmin, _vmax = None, None

    fig, ax = plt.subplots(figsize=figsize);
    sns.heatmap(_cm, annot=True, ax=ax, fmt=_fmt, xticklabels=le.classes_, yticklabels=le.classes_, vmin=_vmin, vmax=_vmax);

    if norm:plt.title(f'Confusion matrix (percentage of class samples) using {classifier_str}')
    else: plt.title(f'Confusion matrix (absolute values) using {classifier_str}')

    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    if save: plt.savefig(f"{classifier_str}_cm_{'norm' if norm else 'abs'}.tiff", dpi=300, bbox_inches='tight')
    plt.show()

def plot_onevsrest_roc(ground_truth, probs, le, save=False, classifier_str=None, figsize=(10, 10)):

    ground_truth_bin = label_binarize(ground_truth, classes=[*range(len(le.classes_))])

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(len(le.classes_)):
        fpr[i], tpr[i], _ = roc_curve(ground_truth_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(len(le.classes_)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= len(le.classes_)

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=figsize)

    auc_dict = {
        'nmd': [],
        'auc': [],
    }

    for class_id in range(len(le.classes_)):    

        c_fpr, c_tpr, _ = roc_curve(
            ground_truth_bin[:, class_id],
            probs[:, class_id]
        )
        c_roc_auc = auc(c_fpr, c_tpr)
        auc_dict['nmd'].append(le.classes_[class_id])
        auc_dict['auc'].append(c_roc_auc)

        RocCurveDisplay.from_predictions(
            ground_truth_bin[:, class_id],
            probs[:, class_id],
            name=f"ROC for {le.classes_[class_id]}",
            color='#005e82',
            alpha=0.5,
            ax=ax,
            linewidth = 1
        )

    _micro_line, = plt.plot(
        fpr["micro"],
        tpr["micro"],
        color="#e61937",
        linestyle="--",
        linewidth=2,
    )

    _macro_line, = plt.plot(
        fpr["macro"],
        tpr["macro"],
        color="#e61937",
        linestyle=":",
        linewidth=2,
    )

    ax.set_aspect('equal', adjustable='box')
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'One-vs-Rest ROC using {classifier_str}')
    plt.legend(
        [_micro_line, _macro_line], 
        [
            f'micro-averaged ROC (AUC = {roc_auc["micro"]:.5f})', 
            f'macro-averaged ROC (AUC = {roc_auc["macro"]:.5f})'
        ]
    )
    if save:
        plt.savefig(f"{classifier_str}_roc.tiff", dpi=300, bbox_inches='tight')
        pd.DataFrame(auc_dict).to_csv(f"{classifier_str}_auc.csv")
    plt.show()

def plot_onevsrest_prc(ground_truth, probs, le, save=False, classifier_str=None, figsize=(10, 10)):

    # Binarize true labels
    ground_truth_bin = label_binarize(ground_truth, classes=[*range(len(le.classes_))])

    # Calculate precision and recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i, c in enumerate(le.classes_):
        precision[i], recall[i], _ = precision_recall_curve(ground_truth_bin[:, i], probs[:, i])
        average_precision[c] = average_precision_score(ground_truth_bin[:, i], probs[:, i])

    # Micro-averaged precision-recall
    precision_micro, recall_micro, _ = precision_recall_curve(ground_truth_bin.ravel(), probs.ravel())
    average_precision_micro = average_precision_score(ground_truth_bin, probs, average="micro")

    # Macro-averaged precision-recall
    precision_macro = np.linspace(0, 1, 1000)
    recall_macro = np.linspace(0, 1, 1000)

    for i in range(len(le.classes_)):
        precision_macro_i, recall_macro_i, _ = precision_recall_curve(ground_truth_bin[:, i], probs[:, i])
        precision_macro += np.interp(recall_macro, recall_macro_i[::-1], precision_macro_i[::-1])
            
    precision_macro /= len(le.classes_)
    average_precision_macro = np.mean([average_precision[c] for c in le.classes_])

    # Plot the precision-recall curves
    plt.figure(figsize=figsize)

    for i, c  in enumerate(le.classes_):
        plt.plot(
            recall[i],
            precision[i],
            color='#005e82',
            alpha=0.5,
            lw=1,
            label=f'_{c} (AUPRC = {average_precision[c]:.2f})', # Underscore to hide from legend
        )

    _micro_line, =plt.plot(
        recall_micro,
        precision_micro,
        color="#e61937",
        linestyle="--",
        linewidth=2,
    )

    _macro_line, =plt.plot(
        recall_macro,
        precision_macro,
        color="#e61937",
        linestyle=":",
        linewidth=2,
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'One-vs-Rest precision-recall curve using {classifier_str}')
    plt.legend(
        [_micro_line, _macro_line], 
        [
            f'micro-averaged PRC (AUPRC = {average_precision_micro:.5f})', 
            f'macro-averaged PRC (AUPRC = {average_precision_macro:.5f})'
        ],
        loc='best'
    )
    if save:
        plt.savefig(f"{classifier_str}_pr_curve.tiff", dpi=300, bbox_inches='tight')
        pd.DataFrame.from_dict(average_precision, orient='index').to_csv(f"{classifier_str}_pr_curve-ap.csv")
    plt.show()

def plot_onevsone_prc(ground_truth, probs, le, X, y, mode='all', save=False, classifier_str=None, figsize=(15, 15)):
    if mode not in ['all', 'roc', 'prc']: raise Exception('`mode` must be "all", "roc" or "prc".')

    n_classes = len(le.classes_)
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    pair_list = list(combinations(list(range(n_classes)), 2))
    # I don't remember what this was supposed to do
    # ovo_tpr = np.zeros_like(fpr_grid)
    mean_mercuri_score = X['mean']

    idx_a_list, idx_b_list, mean_tpr_list, recall_list,  = [], [], [], []
    precision_list, mean_score_list, ap_list = [], [], []

    # Calculate curves
    for ix, (idx_a, idx_b) in enumerate(pair_list):
        idx_a_list.append(idx_a)
        idx_b_list.append(idx_b)

        a_mask = ground_truth == idx_a
        b_mask = ground_truth == idx_b
        ab_mask = np.logical_or(a_mask, b_mask)
        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]
        fpr_a, tpr_a, _ = roc_curve(a_true, probs[ab_mask, idx_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, probs[ab_mask, idx_b])

        ##### ROC calc #####
        if mode in ['all', 'roc']:
            mean_tpr = np.zeros_like(fpr_grid)
            mean_tpr += np.interp(fpr_grid, fpr_a, tpr_a)
            mean_tpr += np.interp(fpr_grid, fpr_b, tpr_b)
            mean_tpr /= 2
            mean_score = auc(fpr_grid, mean_tpr)
            # I don't remember what this was supposed to do
            # ovo_tpr += mean_tpr
            mean_tpr_list.append(mean_tpr)
            mean_score_list.append(mean_score)

        ##### PRC calc #####
        if mode in ['all', 'prc']:
            precision, recall, _ = precision_recall_curve(a_true, probs[ab_mask, idx_a])
            ap = average_precision_score(a_true, probs[ab_mask, idx_a])

            recall_list.append(recall)
            precision_list.append(precision)
            ap_list.append(ap)

    # Define Colormap
    if mode == 'all':
        _scores_merged = mean_score_list.copy()
        _scores_merged.extend(ap_list)
    elif mode == 'roc':
        _scores_merged = mean_score_list.copy()
    elif mode == 'prc':
        _scores_merged = ap_list.copy()

    norm = mpl.colors.Normalize(
        vmin=np.min(_scores_merged), 
        vmax=np.max(_scores_merged)
    )
    colors = [
        (234/255, 60/255, 60/255),
        (206/255, 87/255, 37/255),
        (195/255, 150/255, 31/255),
        (123/255, 198/255, 25/255),
        (25/255, 168/255, 68/255),
        (13/255, 132/255, 123/255),
        (0, 109/255, 168/255),
    ]
    cmap_name = 'cool_warm_light'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    def _draw_roc(axs, idx_a, idx_b, ix):
        ax = axs[idx_a][idx_b]
        ax.plot(
            fpr_grid,
            mean_tpr_list[ix],
            #label=f"Mean {idx_a} vs {idx_b} (AUC = {pair_scores[ix]:.2f})",
            label=f"_Mean {idx_a} vs {idx_b}",
            color='k'
        )
        ax.plot([0, 1], [0, 1], color='#423738', linestyle='--')
        ax.set_facecolor(cmap(norm(mean_score_list[ix])))

    def _draw_prc(axs, idx_a, idx_b, ix):
        ax = axs[idx_b][idx_a]
        ax.plot(
            recall_list[ix],
            precision_list[ix],
            lw=1,
            color='k'
        )
        ax.set_facecolor(cmap(norm(ap_list[ix])))

    # Draw plot
    fig, axs = plt.subplots(ncols=n_classes, nrows=n_classes, figsize=figsize, sharex=False, sharey=False)
    for ix, _ in enumerate(pair_list):
        # Diagonal Histogram
        if ix < n_classes:
            ax = axs[ix][ix]
            sns.histplot(x=mean_mercuri_score[y == ix], bins=15, ax=ax, stat='density', color='#fff', edgecolor='none');
            ax.set_facecolor('#000')

        idx_a = idx_a_list[ix]
        idx_b = idx_b_list[ix]

        # ROC plot
        if mode == 'all':
            _draw_roc(axs, idx_a, idx_b, ix)
        elif mode == 'roc':
            _draw_roc(axs, idx_a, idx_b, ix)
            _draw_roc(axs, idx_b, idx_a, ix)

        #PRC plot
        if mode == 'all':
            _draw_prc(axs, idx_a, idx_b, ix)
        elif mode == 'prc':
            _draw_prc(axs, idx_a, idx_b, ix)
            _draw_prc(axs, idx_b, idx_a, ix)

    # I don't remember what this was supposed to do
    # ovo_tpr /= sum(1 for pair in enumerate(pair_list))
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(None)
        ax.set_xlabel(None)

    for c in range(n_classes):
        axs[0][c].set_title(le.classes_[c], fontsize=12)
        axs[c][0].set_ylabel(le.classes_[c], rotation=90, fontsize=12)

    m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    cax=axs[n_classes-1][n_classes-1].inset_axes([1.1, 0, .7, n_classes])
    cb = plt.colorbar(m, cax=cax)
    existing_ticks = cb.get_ticks()
    existing_ticks[0] = np.min(_scores_merged)
    existing_ticks[-1] = np.max(_scores_merged)
    cb.set_ticks(existing_ticks)

    if mode == 'all':
        cb.set_label('AP and AUC')
        fig.suptitle(f'One-vs-One ROCs and PRCs using a {classifier_str} ensemble', verticalalignment='top', y=0.95) # Add noise to title
    elif mode == 'roc':
        cb.set_label('AUC')
        fig.suptitle(f'One-vs-One ROCs using a {classifier_str} ensemble', verticalalignment='top', y=0.93)
    elif mode == 'prc':
        cb.set_label('AUPRC')
        fig.suptitle(f'One-vs-One PRCs using a {classifier_str} ensemble', verticalalignment='top', y=0.93)

    fig.tight_layout()
    plt.subplots_adjust(hspace=.0, wspace=0.)
    if save:
        plt.savefig(f"{classifier_str}_onevsone_{mode}.tiff", dpi=300, bbox_inches='tight')
    plt.show()
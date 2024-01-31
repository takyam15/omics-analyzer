import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa

sns.set_context('poster')


class OmicsAnalyzer:
    """
    Executes data processing for omics data.

    Attributes
    ----------
    raw_results : pandas.DataFrame
        Original table of data matrix
    processed_results : pandas.DataFrame
        Updated table of data matrix
    features : pandas.DataFrame
        Table of measured features
    samples : pandas.DataFrame
        Table of measured samples
    str_mean : str
        The string which stands for "average"
        This is mainly used for the column name of generated tables.
    str_sd : str
        The string which stands for "standard deviation"
        This is mainly used for the column name of generated tables.
    str_cv : str
        The string which stands for "coefficient of variation"
        This is mainly used for the column name of generated tables.
    str_fc : str
        The string which stands for "fold change"
        This is mainly used for the column name of generated tables.
    str_pval : str
        The string which stands for "p value" caalculated in t-test
        This is mainly used for the column name of generated tables.
    str_qval : str
        The string which stands for "q value" as a corrected p value
        This is mainly used for the column name of generated tables.
    str_log2_fc : str
        The string which stands for "log2(fold change)"
        This is mainly used for the column name of generated tables.
    str_minus_log10_pval : str
        The string which stands for "-log10(p value)"
        This is mainly used for the column name of generated tables.
    str_is_significant : str
        The string which stands for "is significant"
        This is mainly used for the column name of generated tables.
    str_pca_axis : str
        The string for the name of principal components
        This is mainly used for the axis label of PCA plots.
    str_pls_axis : str
        The string for the name of principal components of PLS
        This is mainly used for the the axis label of PLS plots.
    str_score : str
        The string which stands for "score"
        This is mainly used for the key of generated dictionaries.
    str_loading : str
        The string which stands for "loading"
        This is mainly used for the key of generated dictionaries.
    str_vr : str
        The string which stands for "explained variance ratio"
        This is mainly used for the key of generated dictionaries.
    str_vip : str
        The string which stands for "VIP"
        This is mainly used for the column name of generated tables.
    """

    def __init__(self, excel, result_sheet, feature_sheet, sample_sheet):
        """
        Parameters
        ----------
        excel : str
            The imported excel book name (without file extension)
        result_sheet : str
            The name of the sheet in which data matrix of analysis results
            is stored.
        feature_sheet : str
            The name of the sheet in which information of measured features
            is stored.
        sample_sheet : str
            The name of the sheet in which information of measured samples
            is stored.
        """
        self.raw_results = pd.read_excel(
            excel, sheet_name=result_sheet, index_col=0, engine='openpyxl'
        )
        self.processed_results = self.raw_results.copy()
        self.features = pd.read_excel(
            excel, sheet_name=feature_sheet, index_col=0, engine='openpyxl'
        )
        self.samples = pd.read_excel(
            excel, sheet_name=sample_sheet, index_col=0, engine='openpyxl'
        )
        self.str_mean = 'mean'
        self.str_sd = 'sd'
        self.str_cv = 'cv'
        self.str_fc = 'fold-change'
        self.str_pval = 'p-value'
        self.str_qval = 'q-value'
        self.str_log2_fc = 'log2_fc'
        self.str_minus_log10_pval = '-log10_pval'
        self.str_is_significant = 'is_significant'
        self.str_pca_axis = 'PC'
        self.str_pls_axis = 'PLS'
        self.str_score = 'score'
        self.str_loading = 'loading'
        self.str_vr = 'variance_ratio'
        self.str_vip = 'vip'

    def impute_missing_values(self, min_fill=0.5, coef=0.5):
        """
        Parameters
        ----------
        min_fill : float, default 0.5

        coef : float, default 0.5
            between 0 and 1
        """
        self.processed_results.fillna(0, inplace=True)
        detected_features = []

        for feature in self.processed_results.index:
            detected = False
            detected_in_any_samples = (
                self.processed_results.loc[feature] != 0
            ).any()

            if detected_in_any_samples:
                min_val = self.processed_results.loc[
                    feature, self.processed_results.loc[feature] != 0
                ].min()
            else:
                min_val = 0

            for group in self.samples.iloc[:, 0].unique():
                samples = self.samples[self.samples.iloc[:, 0] == group].index
                num_samples = (self.samples.iloc[:, 0] == group).sum()
                extracted_results = self.processed_results.loc[
                    feature, samples
                ]
                num_detected = (extracted_results != 0).sum()

                if num_detected / num_samples >= min_fill:
                    detected = True

                not_detected_samples = extracted_results[
                    extracted_results == 0
                ].index
                self.processed_results.loc[
                    feature, not_detected_samples
                ] = coef * min_val

            if detected:
                detected_features.append(feature)

        self.processed_results = self.processed_results.loc[detected_features]

    def calculate_num_rows_and_cols(self, n_plots, n_cols):
        """
        Calculates number of rows and columns for subplots.

        Parameters
        ----------
        n_plots : int
            Number of subplots in the figure
        n_cols : int
            Number of columns in the figure

        Returns
        -------

        """
        n_rows = n_plots // n_cols

        if n_plots % n_cols:
            n_rows += 1

        if n_plots < n_cols:
            n_cols = n_plots

        if n_rows * n_cols == 0:
            n_rows, n_cols = 1, 1

        return n_rows, n_cols

    def get_ax(self, axes, i, n_rows, n_cols):
        """
        Gets axis object for the subplot.

        Parameters
        ----------
        axes

        i : int

        n_rows : int
            Number of rows in the figure
        n_cols : int
            Number of columns in the figure

        Returns
        -------

        """

        if n_rows > 1 and n_cols > 1:
            return axes[i // n_cols, i % n_cols]
        elif n_rows * n_cols == 1 or n_rows * n_cols == 0:
            return axes
        else:
            return axes[i]

    def show_barplots(
        self, features, groups,
        n_cols=5, row_size=15, col_size=15,
        ci='sd', hue=None, show_stripplots=True, plot_size=10
    ):
        """
        Shows barplots of the targeted features in the targeted sample groups.

        Parameters
        ----------
        features : list
            Targeted features
        groups : list
            Sample groups shown in the plot
        n_cols : int, default 5
            Number of columns in the figure
        row_size : float, default 15
            Length of the vertical axis of a subplot
        col_size : float, default 15
            Length of the horizontal axis of a subplot
        ci : str, default 'sd'

        hue : str or None, default None

        show_stripplots : bool, default True
            if True, stripplots are overlaid.
        plot_size : float, default 10
            Point size of stripplots

        Returns
        -------

        """
        col_group = self.samples.columns[0]
        target_samples = self.samples[
            self.samples[col_group].isin(groups)
        ].index
        df_plot = self.samples.join(
            self.processed_results.loc[features, target_samples].transpose(),
            how='right'
        )
        n_rows, n_cols = self.calculate_num_rows_and_cols(
            len(features), n_cols
        )
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(col_size*n_cols, row_size*n_rows), tight_layout=True
        )

        for i, feature in enumerate(features):
            ax_i = self.get_ax(axes, i, n_rows, n_cols)
            sns.barplot(
                data=df_plot, x=col_group, y=feature,
                ci=ci, hue=hue, capsize=0.2,
                ax=ax_i
            )

            if show_stripplots:
                sns.stripplot(
                    data=df_plot, x=col_group, y=feature,
                    jitter=True, dodge=True, color='black', size=plot_size,
                    ax=ax_i
                )

            ax_i.set(
                xlabel='',
                ylabel='Abundance'
            )
            ax_i.set_title(feature, fontsize=60)
            ax_i.set_xticklabels(
                ax_i.get_xticklabels(), rotation=45, ha='right'
            )

        return fig

    def show_boxplots(
        self, features, groups,
        n_cols=1, row_size=15, col_size=15,
        hue=None, show_stripplots=True, plot_size=10
    ):
        """
        Shows boxplots of the targeted features in the targeted sample groups.

        Parameters
        ----------
        features : list

        groups : list

        n_cols : int, default 1

        row_size : float, deafult 15

        col_size : float, default 15

        hue : str or None, default None

        show_stripplots : bool, default True

        plot_size : float, default 10

        Returns
        -------

        """
        col_group = self.samples.columns[0]
        target_samples = self.samples[
            self.samples[col_group].isin(groups)
        ].index
        df_plot = self.samples.join(
            self.processed_results.loc[features, target_samples].transpose(),
            how='right'
        )
        n_rows, n_cols = self.calculate_num_rows_and_cols(
            len(features), n_cols
        )
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(col_size*n_cols, row_size*n_rows), tight_layout=True
        )

        for i, feature in enumerate(features):
            ax_i = self.get_ax(axes, i, n_rows, n_cols)

            if show_stripplots:
                sns.boxplot(
                    data=df_plot, x=col_group, y=feature,
                    hue=hue, showfliers=False,
                    ax=ax_i
                )
                sns.stripplot(
                    data=df_plot, x=col_group, y=feature,
                    jitter=True, dodge=True, color='black', size=plot_size,
                    ax=ax_i
                )
            else:
                sns.boxplot(
                    data=df_plot, x=col_group, y=feature,
                    hue=hue, showfliers=True,
                    ax=ax_i
                )

            ax_i.set(
                xlabel='',
                ylabel='Abundance'
            )
            ax_i.set_title(feature, fontsize=60)
            ax_i.set_xticklabels(
                ax_i.get_xticklabels(), rotation=45, ha='right'
            )

        if hue is not None:
            n_hue = df_plot[hue].unique().count()
            handles, labels = ax_i.get_legend_handles_labels()
            ax_i.legend(handles[0:n_hue], labels[0:n_hue])

        return fig

    def calculate_fold_change_and_pvalue(
        self, groups, equal_var, alpha, fdr_corr_method
    ):
        """
        Parameters
        ----------
        groups : list

        equal_var : bool

        alpha : float

        fdr_corr_method : str

        Returns
        -------

        """
        samples = self.samples.loc[self.samples.iloc[:, 0].isin(groups)].index
        df_results = self.processed_results.loc[:, samples].copy()

        for group in groups:
            group_samples = self.samples.loc[
                self.samples.iloc[:, 0] == group
            ].index
            df_group = df_results.loc[:, group_samples]
            group_mean = df_group.mean(axis=1)
            group_sd = df_group.std(axis=1)
            group_cv = group_sd * 100 / group_mean
            df_results.loc[:, f'{group}_{self.str_mean}'] = group_mean
            df_results.loc[:, f'{group}_{self.str_sd}'] = group_sd
            df_results.loc[:, f'{group}_{self.str_cv}'] = group_cv

        group0_mean = df_results.loc[:, f'{groups[0]}_{self.str_mean}']
        group1_mean = df_results.loc[:, f'{groups[1]}_{self.str_mean}']
        df_results.loc[:, self.str_fc] = group1_mean / group0_mean
        group0_samples = self.samples.loc[
            self.samples.iloc[:, 0] == groups[0]
        ].index
        group1_samples = self.samples.loc[
            self.samples.iloc[:, 0] == groups[1]
        ].index

        for feature in df_results.index:
            df_results.loc[feature, self.str_pval] = ttest_ind(
                df_results.loc[feature, group0_samples],
                df_results.loc[feature, group1_samples],
                equal_var=equal_var
            ).pvalue

        df_results.loc[:, self.str_qval] = fdrcorrection(
            df_results[self.str_pval], alpha=alpha, method=fdr_corr_method
        )[1]

        return df_results

    def extract_differentially_expressed_features(
        self, groups, equal_var=False, min_fc=1.2, max_pval=0.05,
        fdr_corr_method='indep', use_qval=True
    ):
        """
        Parameters
        ----------
        groups : list

        equal_var : bool, default False

        min_fc : float, default 1.2

        max_pval : float, default 0.05

        fdr_corr_method : str, default 'indep'

        use_qval : bool, default False

        Returns
        -------

        """
        df_results = self.calculate_fold_change_and_pvalue(
            groups, equal_var, max_pval, fdr_corr_method
        )
        df_results.loc[:, self.str_is_significant] = 0

        if use_qval:
            col_pval = self.str_qval
        else:
            col_pval = self.str_pval

        fc_is_higher = df_results.loc[:, self.str_pval] >= min_fc
        fc_is_lower = df_results.loc[:, self.str_pval] <= 1 / min_fc
        pval_is_lower = df_results.loc[:, col_pval] < max_pval

        df_results.loc[
            ((fc_is_higher) | (fc_is_lower)) & (pval_is_lower),
            self.str_is_significant
        ] = 1
        self.comparative_results = df_results
        df_significant = df_results.loc[
            df_results.loc[:, self.str_is_significant] == 1
        ]
        return df_significant.drop(self.str_is_significant, axis=1)

    def show_volcano_plot(
        self, groups, equal_var=False, min_fc=1.2, max_pval=0.05,
        fdr_corr_method='indep', use_qval=True
    ):
        """
        Parameters
        ----------
        groups : list

        equal_var : bool, default False

        min_fc : float, default 1.2

        max_pval : float, default 0.05

        fdr_corr_method : str, default 'indep'

        use_qval : bool, default True

        Returns
        -------

        """
        self.extract_differentially_expressed_features(
            groups, equal_var, min_fc, max_pval, fdr_corr_method, use_qval
        )
        df_plot = self.comparative_results
        df_plot.loc[:, self.str_log2_fc] = np.log2(
            df_plot.loc[:, self.str_fc]
        )
        df_plot.loc[:, self.str_minus_log10_pval] = -np.log10(
            df_plot.loc[:, self.str_pval]
        )
        fig, ax = plt.subplots(figsize=(30, 30))
        sns.scatterplot(
            data=df_plot, x=self.str_log2_fc, y=self.str_minus_log10_pval,
            hue=self.str_is_significant, legend=False,
            ax=ax
        )
        return fig

    def normalize(self, groups):
        """
        Parameters
        ----------
        groups : list
        """
        sc = StandardScaler()
        features = self.processed_results.index
        samples = self.samples.loc[self.samples.iloc[:, 0].isin(groups)].index
        normalized_results = sc.fit_transform(
            self.processed_results.loc[features, samples].transpose()
        )
        self.normalized_results = pd.DataFrame(
            normalized_results,
            index=samples,
            columns=features
        ).transpose()

    def show_hca_heatmap(
        self, groups, method='ward', metric='euclidean', cmap='bwr'
    ):  # TODO: col/row_cluster選択できるようにする、col/row_colors指定できるようにする
        """
        Parameters
        ----------
        groups : list

        method : str, default 'ward'

        metrics : str, default 'euclidean'

        cmap : str, default 'bwr'

        Returns
        -------

        """
        self.normalize(groups)
        rows = len(self.normalized_results.index)
        cols = len(self.normalized_results.columns)
        return sns.clustermap(
            data=self.normalized_results, method=method, metric=metric,
            cmap=cmap, center=0,
            figsize=(cols, rows)
        )

    def calculate_pca_scores_and_loadings(self, groups, n_comp):
        """
        Parameters
        ----------
        groups : list

        n_comp : int

        Returns
        -------

        """
        self.normalize(groups)
        X = self.normalized_results.transpose()
        pca = PCA(n_components=n_comp)
        features = X.columns
        samples = X.index
        principal_components = [
            f'{self.str_pca_axis}{i+1}' for i in range(n_comp)
        ]
        df_score = pd.DataFrame(
            pca.fit_transform(X),
            index=samples,
            columns=principal_components
        )
        df_corr = X.join(df_score).corr()
        df_loading = df_corr.loc[features, principal_components]
        evr = pca.explained_variance_ratio_
        str_evr = 'explained_variance_ratio'
        df_variance_ratio = pd.DataFrame(
            evr,
            index=principal_components,
            columns=[str_evr]
        )
        return {
            self.str_score: df_score,
            self.str_loading: df_loading,
            self.str_vr: df_variance_ratio
        }

    def show_pca_score_plots(
        self, groups, n_comp=8, n_cols=2, row_size=15, col_size=15
    ):
        """
        Shows PCA score plots.

        Parameters
        ----------
        groups : list

        n_comp : int, default 8

        n_cols : int, default 2

        row_size : float, default 15

        col_size : float, default 15

        Returns
        -------

        """
        pca_dict = self.calculate_pca_scores_and_loadings(groups, n_comp)
        df_score = pca_dict[self.str_score]
        df_variance_ratio = pca_dict[self.str_vr]
        evr_sum = round(df_variance_ratio.iloc[:, 0].sum() * 100, 1)
        df_plot = self.samples.join(df_score)
        n_plots = n_comp // 2 + n_comp % 2
        n_rows, n_cols = self.calculate_num_rows_and_cols(n_plots, n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(col_size*n_cols, row_size*n_rows)
        )
        principal_components = df_score.columns
        col_group = df_plot.columns[0]

        for i in range(n_plots):
            ax_i = self.get_ax(axes, i, n_rows, n_cols)
            pc_x = principal_components[i * 2]
            pc_y = principal_components[i * 2 + 1]
            evr_x = round(df_variance_ratio.iloc[i * 2, 0] * 100, 1)
            evr_y = round(df_variance_ratio.iloc[i * 2 + 1, 0] * 100, 1)
            sns.scatterplot(
                data=df_plot, x=pc_x, y=pc_y,
                hue=col_group,
                ax=ax_i
            )
            ax_i.set(
                xlabel=f'{pc_x} ({evr_x}%)',
                ylabel=f'{pc_y} ({evr_y}%)'
            )
            lg = ax_i.legend_
            lg.set_title('')

        fig.suptitle(f'{self.str_pca_axis} 1 ~ {n_comp}: {evr_sum}%')
        return fig

    def extract_features_correlated_with_pca_scores(
        self, groups, min_coef=0.7, pc=1, n_comp=8
    ):
        """
        Parameters
        ----------
        groups : list

        min_coef : float, default 0.7

        pc : int, default 1

        n_comp : int, default 8  # TODO: 元データが8特徴量未満の場合どうするか

        Returns
        -------

        """
        pca_dict = self.calculate_pca_scores_and_loadings(groups, n_comp)
        df_loading = pca_dict[self.str_loading]
        col_pc = df_loading.columns[pc-1]
        df_loading.sort_values(col_pc, ascending=False, inplace=True)
        df_loading_pc = df_loading.loc[:, col_pc]
        return df_loading_pc[df_loading_pc >= min_coef]

    def calculate_pls_scores_and_loadings(
        self, groups, cols_numerical_y, cols_categorical_y, max_n_splits
    ):
        """
        Calculates values of PLS score and loading

        Parameters
        ----------
        groups : list

        cols_numerical_y : list

        cols_categorical_y : list

        max_n_splits : int

        Returns
        -------

        """
        self.normalize(groups)
        X = self.normalized_results.transpose()
        features = X.columns
        samples = X.index
        df_samples = self.samples.loc[
            self.samples.iloc[:, 0].isin(groups),
            cols_numerical_y + cols_categorical_y
        ]
        df_y = pd.get_dummies(df_samples, columns=cols_categorical_y)
        sc = StandardScaler()
        y = sc.fit_transform(df_y)
        max_score = 0
        max_n = 2
        n_splits = len(df_samples) // 2

        if n_splits > max_n_splits:
            n_splits = max_n_splits

        pls = PLSRegression(n_components=max_n)
        score = cross_val_score(
            pls, X, y, cv=KFold(n_splits, shuffle=True)
        ).mean()

        while score >= max_score:
            max_n += 1
            max_score = score
            pls = PLSRegression(n_components=max_n)
            score = cross_val_score(
                pls, X, y, cv=KFold(n_splits, shuffle=True)
            ).mean()

        n_comp = max([2, max_n - 1])
        pls = PLSRegression(n_components=n_comp)
        principal_components = [
            f'{self.str_pls_axis}{i+1}' for i in range(n_comp)
        ]
        df_score_x = pd.DataFrame(
            pls.fit_transform(X, y)[0],
            index=samples,
            columns=principal_components
        )
        df_corr = X.join(df_score_x).corr()
        df_loading = df_corr.loc[features, principal_components]
        t = pls.x_scores_
        w = pls.x_weights_
        q = pls.y_loadings_
        p = len(X.columns)
        h = t.shape[1]
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array(
                [(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)]
            )
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

        df_vip = pd.DataFrame(index=X.columns)
        col_vip = 'VIP'
        df_vip.loc[:, col_vip] = vips
        return {
            self.str_score: df_score_x,
            self.str_loading: df_loading,
            self.str_vip: df_vip
        }

    def show_pls_score_plots(
        self, groups, cols_numerical_y, cols_categorical_y, max_n_splits=7,
        n_cols=2, row_size=15, col_size=15
    ):
        """
        Shows PLS score plots.

        Parameters
        ----------
        groups : list

        cols_numerical_y : list of str

        cols_categorical_y : list of str

        max_n_splits : int, default 7

        n_cols : int, default 2

        row_size : float, default 15

        col_size : float, default 15

        Returns
        -------

        """
        pls_dict = self.calculate_pls_scores_and_loadings(
            groups, cols_numerical_y, cols_categorical_y, max_n_splits
        )
        df_score = pls_dict[self.str_score]
        n_comp = min(len(groups)-1, len(df_score.columns))
        df_plot = self.samples.join(df_score, how='right')
        n_plots = n_comp // 2 + n_comp % 2
        n_rows, n_cols = self.calculate_num_rows_and_cols(n_plots, n_cols)

        if n_comp == 1:
            col_group = df_plot.columns[0]
            col_pls1 = f'{self.str_pls_axis}1'
            df_plot.reset_index(inplace=True)
            col_samples = df_plot.columns[0]
            fig, ax = plt.subplots(figsize=(col_size*2, row_size))
            sns.barplot(
                data=df_plot, x=col_samples, y=col_pls1,
                hue=col_group,
                ax=ax
            )
            ax.set(
                xlabel=''
            )
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha='right'
            )
            lg = ax.legend_
            lg.set_title('')
        else:
            fig, axes = plt.subplots(
                figsize=(col_size*n_cols, row_size*n_rows)
            )

            for i in range(n_plots):
                ax_i = self.get_ax(axes, i, n_rows, n_cols)
                pls_x = df_score.columns[i * 2]
                pls_y = df_score.columns[i * 2 + 1]
                col_group = df_plot.columns[0]
                sns.scatterplot(
                    data=df_plot, x=pls_x, y=pls_y,
                    hue=col_group,
                    ax=ax_i
                )
                ax_i.set(
                    xlabel=pls_x,
                    ylabel=pls_y
                )
                lg = ax_i.legend_
                lg.set_title('')

        return fig

    def extract_features_correlated_with_pls_scores(
        self, groups, cols_numerical_y, cols_categorical_y,
        max_n_splits=7, min_coef=0.7, pc=1
    ):
        """
        Extracts features highly correlated with PLS scores.

        Parameters
        ----------
        groups : list

        cols_numerical_y : list

        cols_categorical_y : list

        max_n_splits : int, default 7

        min_coef : float, default 0.7

        pc : int, default 1

        Returns
        -------

        """
        pls_dict = self.calculate_pls_scores_and_loadings(
            groups, cols_numerical_y, cols_categorical_y, max_n_splits
        )
        df_loading = pls_dict[self.str_loading]
        col_pls = f'{self.str_pls_axis}{pc}'
        df_loading.sort_values(col_pls, ascending=False, inplace=True)
        df_loading_pc = df_loading.loc[:, col_pls]
        return df_loading_pc[df_loading_pc >= min_coef]

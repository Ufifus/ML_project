import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


width, height = 1200, 800

"""Столбчатая диаграма на распределение элементов по значению"""
class Histogram:
    def __init__(self):
        """"""
        self.configs = {}

    def get_configs(self, configs_):
        """"""
        if configs_['type'] == 0:
            print('nice')
            self.configs['category'] = True
        else:
            self.configs['category'] = False

    def plot(self, data, label, others):
        if others is None:
            if self.configs['category']:
                fig = px.histogram(data, x=label, category_orders=data[label].unique().tolist(), width=width, height=height)
            else:
                fig = px.histogram(data, x=label, width=width, height=height)
        else:
            fig = px.histogram(data, x=label, color=others, width=width, height=height)

        return fig


class Scatter:
    def __init__(self):
        """"""

    def get_config(self):
        """"""

    def plot(self, data, label, others):
        if others:
            fig = px.scatter(data, x=label[0], y=label[1], color=others, width=width, height=height,
                             marginal_x="violin", marginal_y="box")
        else:
            fig = px.scatter(data, x=label[0], y=label[1], width=width, height=height,
                             marginal_x="histogram", marginal_y="rug")
        return fig


class Bubbles:
    def __init__(self):
        """"""

    def get_configs(self):
        """"""

    def plot(self, data, label, bubble, hover, others):
        print(bubble, others)
        if others is not None:
            fig = px.scatter(data, x=label[0], y=label[1], color=others, width=width, height=height,
                             size=bubble, size_max=60, log_x=True, hover_name=hover)
        else:
            fig = px.scatter(data, x=label[0], y=label[1], width=width, height=height,
                             size=bubble, size_max=60, log_x=True, hover_name=hover)
        return fig


class Heapmap:
    def __init__(self):
        """"""

    def get_config(self):
        """"""

    def plot(self, data, useable_labels):

        data = data[useable_labels].corr().round(2)
        fig = px.imshow(data, labels=dict(x='Корреляционная матрица признаков', color='близость'),
                        x=useable_labels, y=useable_labels, width=width, height=height,
                        text_auto=True)
        return fig

    def plot_not_pd(self, data, labels):

        fig = px.imshow(data, labels=dict(x='Accuracy', color='Accuracy'),
                        x=labels, y=labels, width=width, height=height,
                        text_auto=True)
        return fig


class NaTable:
    def __init__(self):
        """"""

    def get_config(self):
        """"""

    def plot(self, data):
        count = data.isna().sum().to_frame()
        print('---')
        print(count)
        print('---')
        fig = px.imshow(count, labels=dict(x='Table of clear slots', color='Does not exist'),
                        text_auto=True, width=width, height=height)
        return fig


class ROC_Curve:
    def __init__(self):
        """"""

    def plot(self, y_scores, y_onehot):
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=width,
            height=height
        )
        return fig


class Multy_Ploting:
    def __init__(self):
        """"""

    def plot_scatter(self, dtf, labels, color):
        fig = px.scatter_matrix(dtf,
                                dimensions=labels,
                                color=color,
                                width=width,
                                height=height)

        return fig

    def plot_diagram(self, dtf, labels, color):
        fig = px.parallel_categories(dtf,
                                     dimensions=labels,
                                     color=color,
                                     width=width,
                                     height=height)
        return fig

    def plot_coordinates(self, dtf, labels, color):
        fig = px.parallel_coordinates(dtf,
                                      dimensions=labels,
                                      color=color,
                                      width=width,
                                      height=height,
                                      color_continuous_scale=px.colors.diverging.Tealrose,
                                      color_continuous_midpoint=2)
        return fig

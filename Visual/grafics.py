import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


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
                fig = px.histogram(data, x=label, category_orders=data[label].unique().tolist())
            else:
                fig = px.histogram(data, x=label)
        else:
            fig = px.histogram(data, x=label, color=others)

        return fig


class Scatter:
    def __init__(self):
        """"""

    def get_config(self):
        """"""

    def plot(self, data, label, others):
        if others:
            fig = px.scatter(data, x=label[0], y=label[1], color=others)
        else:
            fig = px.scatter(data, x=label[0], y=label[1])
        return fig


class Heapmap:
    def __init__(self):
        """"""

    def get_config(self):
        """"""

    def plot(self, data, useable_labels):

        data = data[useable_labels].corr()
        fig = px.imshow(data, labels=dict(x='Корреляционная матрица признаков', color='близость'),
                        x=useable_labels, y=useable_labels,
                        text_auto=True)
        return fig

    def plot_not_pd(self, data, labels):

        fig = px.imshow(data, labels=dict(x='Accuracy', color='Accuracy'),
                        x=labels, y=labels,
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
                        text_auto=True)
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
        )
        return fig
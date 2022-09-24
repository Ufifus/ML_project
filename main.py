import streamlit as st
import pandas as pd

import Visual
import Models

formats = ['csv', 'xlsx']


@st.cache
def load_data(file, file_name):
    format = file_name.split('.')[-1].lower()
    if format == formats[0]:
        data = pd.read_csv(file)
    elif format == formats[1]:
        data = pd.read_excel(file)
    else:
        raise Exception('Format is not available', file_name)
    return data


def sort_data(data, labels=None):
    if labels is None:
        labels = {}
        for label, uniqval in zip(data.columns, data.nunique()):
            info = {}
            value = data[label][0]
            try:
                int(value)
                info['type'] = 1
            except:
                try:
                    float(value)
                    info['type'] = 1
                except:
                    info['type'] = 0
            finally:
                info['uniq_vals'] = uniqval
                info['use'] = True
                labels[label] = info
        for label, nadata in zip(data.columns, data.isna().sum()):
            labels[label]['na_data'] = nadata
    else:
        for label, uniqval in zip(data.columns, data.nunique()):
            labels[label]['uniq_vals'] = uniqval
        for label, nadata in zip(data.columns, data.isna().sum()):
            labels[label]['na_data'] = nadata
    labels['ID']['use'] = False
    return labels


def plot_grafic(name, **kwargs):
    if name == 'Histogram':
        if kwargs['label'] == kwargs['others']:
            st.write('Please choise various labels')
            return None
        hist = Visual.Histogram()
        hist.get_configs(kwargs['configs'])
        fig = hist.plot(kwargs['data'], kwargs['label'], kwargs['others'])
    if name == 'Scatter':
        if kwargs['label'][0] == kwargs['label'][1]:
            st.write('Please choise various labels')
            return None
        hist = Visual.Scatter()
        hist.get_config()
        fig = hist.plot(kwargs['data'], kwargs['label'], kwargs['others'])
    if name == 'Heapmap':
        heap = Visual.Heapmap()
        fig = heap.plot(kwargs['data'], [label for label in st.session_state.labels \
                                         if st.session_state.labels[label]['use'] == True \
                                         and st.session_state.labels[label]['type'] == 1])

    st.plotly_chart(fig, use_container_width=True)
    return None


def update_labels():
    for label in st.session_state.labels:
        if label in st.session_state.new_labels1:
            st.session_state.labels[label]['type'] = 1
        elif label in st.session_state.new_labels2:
            st.session_state.labels[label]['type'] = 0
        else:
            st.session_state.labels[label]['use'] = False
    print(st.session_state.labels)
    return None


def update_params():
    return None


def delete_rows(delete_labels, change_labels):
    print(delete_labels)
    st.session_state.data = st.session_state.data.dropna(subset=delete_labels)
    for label in change_labels:
        if st.session_state.labels[label]['type'] == 1:
            st.session_state.data[label].fillna(st.session_state.data[label].median(), inplace=True)
        if st.session_state.labels[label]['type'] == 0:
            st.session_state.data[label].fillna(st.session_state.data[label].mode().values[0], inplace=True)
    st.session_state.labels = sort_data(st.session_state.data, st.session_state.labels)

    natable = Visual.NaTable()
    fig = natable.plot(st.session_state.data)
    st.plotly_chart(fig, use_container_width=True)


def train_pred(data, y_label, models):
    if y_label is not None:
        data_y = data[y_label].copy()
        st.session_state.labels[y_label]['use'] = False
        X_train, y_train, X_test, y_test = Models.preprocess_data(data, data_y, st.session_state.labels)
        st.session_state.labels[y_label]['use'] = True
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        for model in models:
            st.write(model)
            cm_model, test_accuracy, train_accuracy, y_onehot, y_scores, table_accuracy = Models.trainer(X_train, y_train, X_test, y_test, model)
            st.write('Train accuracy = ', round(train_accuracy, 2))
            st.write('Test accuracy = ', round(test_accuracy, 2))
            heapmap_acc = Visual.Heapmap()
            fig = heapmap_acc.plot_not_pd(cm_model, data_y.unique().tolist())
            st.plotly_chart(fig, use_container_width=True)
            ROC_curve = Visual.ROC_Curve()
            fig = ROC_curve.plot(y_scores, y_onehot)
            st.plotly_chart(fig, use_container_width=True)
            st.write('-----------------------------------')


def init_data():
    st.session_state.data = None


if __name__ == '__main__':
    st.title('ML_testing')
    file = st.file_uploader('Upload data', on_change=init_data())
    if file:
        file_name = file.name
        if 'data' not in st.session_state:
            st.session_state.data = load_data(file, file_name)
        if st.session_state.data is None:
            st.session_state.data = load_data(file, file_name)
        st.dataframe(st.session_state.data.head(20))

        labels = sort_data(st.session_state.data)
        if 'labels' not in st.session_state:
            st.session_state.labels = labels

        with st.form(key='init_data'):
            number_labels = st.multiselect("Number data", st.session_state.data.columns,
                                           [label for label in st.session_state.data.columns if labels[label]['type'] == 1 and \
                                            labels[label]['use'] == True],
                                           key='new_labels1')
            categorial_labels = st.multiselect("Category data", st.session_state.data.columns,
                                               [label for label in st.session_state.data.columns if labels[label]['type'] == 0 and \
                                                labels[label]['use'] == True],
                                               key='new_labels2')
            st.form_submit_button(label='Update', on_click=update_labels())

        grafic = st.selectbox('Choise type of grafic', ['Histogram', 'Scatter', 'Heapmap'],
                              on_change=update_params)

        # создаем форму с выбором полей и видом графика для визуализации данных
        with st.form(key='visual_data'):
            if grafic:
                print(grafic)
                if grafic == 'Histogram':
                    label = st.selectbox('Choise labels', st.session_state.data.columns)
                    other_label = st.selectbox('Choise other categorial param', [None] + [x for x in st.session_state.data.columns])
                    configs = labels[label]
                elif grafic == 'Scatter':
                    label_x = st.selectbox('Choise x label', st.session_state.data.columns)
                    label_y = st.selectbox('Choise y label', st.session_state.data.columns)
                    other_label = st.selectbox('Choise other categorial param', [None] + [x for x in st.session_state.data.columns])
                    label = (label_x, label_y)
                    configs = (labels[label_x], labels[label_y])
                else:
                    label = None
                    other_label = None
                    configs = None

                params = dict(data=st.session_state.data, label=label, others=other_label, configs=configs)
                st.form_submit_button('Ploting', on_click=plot_grafic(grafic, **params))

        with st.form(key='data_clear'):
            st.write('Clear rows in table if exists None elements')
            delete_choices = st.multiselect('Choice that row with clear slots delete',
                                            [label for label in st.session_state.labels if labels[label]['na_data'] != 0 and \
                                             labels[label]['use'] == True])
            change_choices = st.multiselect('Choice that row with clear slots fill',
                                            [label for label in st.session_state.labels if labels[label]['na_data'] != 0 and \
                                             labels[label]['use'] == True])
            st.form_submit_button('Clear or change data', on_click=delete_rows(delete_choices, change_choices))

        with st.form(key='Ml_model'):
            y_label = st.selectbox('Choice label for predict', [None] + [label for label in st.session_state.data.columns])
            model_label = st.multiselect('Choise able models', [model for model in Models.models])
            train = st.form_submit_button('Train', train_pred(st.session_state.data, y_label, model_label))

        print(st.session_state.data)





import pandas as pd
from data_processing.data_cleaning import clean_row_with_null_values
from data_processing.data_convert import convert_all_columns_in_list_to_numeric
from data_visualization.data_graphic import generated_graphic_only_numeric_column
# from ydata_profiling import ProfileReport
# from pandas_profiling import ProfileReport

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

import pickle

# dados_df = pd.read_csv('./data/marketing_investimento.csv')
dados_df = pd.read_csv('./data/clean_marketing_investimento.csv')


def clean_data():
    is_null_values = dados_df.isnull().sum()
    print(is_null_values)
    clean_data_df = clean_row_with_null_values(dados_df, 'aderencia_investimento')
    all_columns_to_convert = ['idade', 'tempo_ult_contato', 'numero_contatos']
    converted_numeric_df = convert_all_columns_in_list_to_numeric(clean_data_df, all_columns_to_convert)
    generated_graphic(converted_numeric_df)

def generated_graphic(converted_numeric_df):
    graphic = generated_graphic_only_numeric_column(converted_numeric_df, 'idade')
    graphic.show()
    

# def generated_data_profilling():
    # profile = ydata_profiling(dados_df)
    # profile.to_file("./data/Report_profile_relatorio_clean.html")
    # clean_data_df.to_csv('./data/clean_marketing_investimento.csv', index=False)


x = dados_df.drop('aderencia_investimento', axis=1)
y = dados_df['aderencia_investimento']

columns_name = x.columns

one_hot = make_column_transformer((
    OneHotEncoder(drop='if_binary'),
    ['estado_civil', 'escolaridade', 'inadimplencia', 'fez_emprestimo']),
    remainder = 'passthrough',
    sparse_threshold=0
)

x = one_hot.fit_transform(x)
get_name_after_transformation = one_hot.get_feature_names_out(columns_name)

pd.DataFrame(x, columns = get_name_after_transformation)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=5)

# Dummy
dummy_classifier = DummyClassifier()
dummy_classifier.fit(x_train, y_train)
score_dummy_model = dummy_classifier.score(x_test, y_test)
print("SCORE DO MODELO DUMMY")
print(score_dummy_model)

# Decision Tree
tree = DecisionTreeClassifier(random_state=5)
tree.fit(x_train, y_train)
score_tree_decision_model = tree.score(x_test, y_test)
print("SCORE DO MODELO ARVORE")
print(score_tree_decision_model)

all_collumns_name = ['casado (a)',
                'divorciado (a)',
                'solteiro (a)',
                'fundamental',
                'medio',
                'superior',
                'inadimplencia',
                'fez_emprestimo',
                'idade',
                'saldo',
                'tempo_ult_contato',
                'numero_contatos']

plt.figure(figsize = (15, 6))
plot_tree(tree, filled = True, class_names = ['nao', 'sim'], fontsize = 1, feature_names = all_collumns_name)
# plt.show()


# KNN
normalization = MinMaxScaler()
x_train_normalize = normalization.fit_transform(x_train)

pd.DataFrame(x_train_normalize)

knn = KNeighborsClassifier()
knn.fit(x_train_normalize, y_train)
x_test_normalize = normalization.transform(x_test)
score_knn_model = knn.score(x_test_normalize, y_test)
print("SCORE DO KNN")
print(score_knn_model)


with open('modelo_arvore.pkl', 'wb')as arquivo:
    pickle.dump(tree, arquivo)

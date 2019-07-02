# %load load_dataset.py

from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto"
                                                     "-mpg/auto-mpg.data")

column_names = ['Расход топлива', 'Кол-во цилиндров', 'Объем двигателя', 'Л.с.', 'Вес', 'Разгон до 100 км/ч',
                'Год выпуска', 'Страна выпуска']

raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

# Common data
dataset = raw_dataset.copy()
dataset = dataset.dropna()

# Normalizing Country
origin = dataset.pop('Страна выпуска')
dataset['США'] = (origin == 1) * 1.0
dataset['Европа'] = (origin == 2) * 1.0
dataset['Япония'] = (origin == 3) * 1.0

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("Расход топлива")
train_stats = train_stats.transpose()


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


train_labels = train_dataset.pop('Расход топлива')
test_labels = test_dataset.pop('Расход топлива')

# Обучающая и тестовая выборка
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def load():
    return normed_train_data, train_labels, normed_test_data, test_labels


# Таблица распределения данных
def data_distribution():
    g = sns.pairplot(train_dataset[["Расход топлива", "Кол-во цилиндров", "Объем двигателя", "Вес"]], diag_kind="kde",
                     kind="reg")
    g.fig.suptitle('Распределение данных')
    plt.show()

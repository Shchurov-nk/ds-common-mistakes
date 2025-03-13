## Популярные сложности и ошибки

### Воспроизводимость
**Нужно работать так, чтобы вы могли в любой момент времени воспроизвести весь проект на компьютере друга за 30 минут, а ваш коллега по лаборатории - за 60 минут**

Указание пути до файла
```python
# Плохо
pd.read_csv("C:/Users/my_project/data/file.csv")

# Лучше, но в .py файле будет зависеть от точки входа (может сломаться)
pd.read_csv("../file.csv")

# Идеально
from settings import data_path
pd.read_csv(data_path)
```
```python
# settings.py
from pathlib import Path
project_path = Path(__file__).resolve().parents[1]
data_path = project_path / "data"
```

Сохранение файлов в неудобных форматах
```python
# Плохо
with open('results.txt', 'a') as f:
    f.write(str_element +'\n')
    for i in range(3):
        r2, mae = fit_my_model(X_trn, Y_trn, X_test, Y_test)
        f.write(str(r2) + '\t' + str(mae) + '\n')
    f.write('\n')
# А теперь руками копируем это в excel

# Лучше
N_REPEATS = 3
accuracy = list()
for i in range(N_REPEATS):
    r2, mae = evaluate_my_model(X_test, Y_test)
    accuracy.append(r2, mae)
acc_df = pd.DataFrame(accuracy, columns=['R²', 'mae'])
acc_df.to_excel('results.xlsx')
```

Сохранение модели и нормировщиков данных
```python
import pickle
model_file = "model.pkl"  

# Сохранение модели
with open(model_file, 'wb') as file:  
    pickle.dump(model, file)

# Загрузка модели
with open(model_file, 'rb') as file:  
    model = pickle.load(file)
```

Циклы вручную
```python
# Плохо
# поочередно комментируем разные строки и запускаем код
# tagret = 'pH'
tagret = 'Cu'
# tagret = 'Ni'

y = df[target]
model.fit(X, y)

# Хорошо
targets = ['pH', 'Cu', 'Ni']
for target in targets:
    y = df[target]
    model.fit(X, y)
```

Работа по памяти  
Куча бекапов в виде копий  
Какие из этих 23 файлов нужно запустить и в какой последовательности?  
Какой из файлов в итоге финальная версия?
```
6_ions_CC_net_one_cell.ipynb
6_ions_FULL_net_one_cell.ipynb
6_ions_IFS_net_one_cell.ipynb
6_ions_net_one_cell.ipynb
6_ions_network-Copy1.ipynb
6_ions_network.ipynb
CFS_mask_2.0-Copy1.ipynb
CFS_mask_2.0.ipynb
fast_masking.ipynb
loading_CC_model_for_clean_data.ipynb
loading_CC_model_for_noisy_data(MASK_HERE).ipynb
loading_CC_model-Copy1.ipynb
loading_CC_model.ipynb
loading_CFS_model_clean_data.ipynb
loading_CFS_model_for_noisy_data.ipynb
loading_CFS_model.ipynb
loading_IFS_model_for_clean_data.ipynb
loading_weights.ipynb
normalization_noisy_data.ipynb
normalization.ipynb
visualization.ipynb
Визуализация_корреляций_XY.ipynb
Вывод_кол-ва_элементов_в_масках.ipynb
```
Лучше использовать git для контроля версий
```bash
git init
git add
git commit
```
И в случае ipynb указывать порядок запуска в названии  
```
0.1-normalization.ipynb
0.2-CFS_mask.ipynb
0.3-visualization.ipynb
```

Неиспользование окружений
```bash
conda create --name my_project python=3.12
conda activate my_project
conda install pandas numpy plotly
```
Неиспользование requirements.txt  
Альтернатива - poetry (сложнее, но больше функционал)
```bash
pip freeze > requirements.txt

pip install -r requirements.txt
# Или через conda
conda install --file requirements.txt
```

### Экономия времени
#### **Код читается чаще, чем пишется**

Код должен быть pythonic
```python
# Плохо
for i in range(len(my_list)):
    print(my_list[i])

# Хорошо
for element in my_list:
    print(element)

# Если нужен порядковый номер
for index, element in enumerate(my_list):
    print(i, element)
```

Использование f-string и list comprehension
```python
# Пример
fruits = ["apple", "banana", "cherry", "mango"]
no_long_names = [f'tasty {i}' for i in fruits if len(i) > 5]

# Еще пример
width, length = 10, 12
my_string = f'A {width} by {length} room has an area of {width * length}.'
```
Излишнее количество комментариев
Лучше переписать код так, чтобы комментарии не требовались
```python
# Не берем первую колонку - там дата
df.iloc[:, 1:] = df.iloc[:, 1:] - 100

# Не нужны комментарии чтобы понять, что происходит
df.set_index('date')
df = df - 100
```
#### Ручной grid-search
Лучше использовать библиотеки для подбора гиперпараметров (optuna, sklearn GridSearchCV)
```python
# TODO: write grid search code example
```

#### Непредвиденные результаты
Неиспользование библиотек, написание функций самостоятельно
```python
# Плохо
def mae(y_true, predictions):
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    return np.mean(np.abs(y_true - predictions)) 

# Хорошо
from sklearn.metrics import mean_absolute_error
```
```python
# Плохо
from keras import backend as K
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Хорошо
from sklearn.metrics import r2_score
```
```python
# Плохо (если вам, конечно, не нужно именно это)
X_train = X.iloc[:80]
X_test = X.iloc[80:]
y_train = y.iloc[:80]
y_test = y.iloc[80:]

# Хорошо
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
    )
```

Константы внутри скрипта
```python
# Плохо
X = df.iloc[:, :2859]
y = df.iloc[:, 2859:]

# Хорошо
COLUMN_SPLIT = 2859
X = df.iloc[:, :COLUMN_SPLIT]
y = df.iloc[:, COLUMN_SPLIT:]

# Хорошо
target_columns = ['pH', 'Cu', 'Ni']
X = df.drop(target_columns, axis=1)
y = df[target_columns]
```

Изменяемые и неизменяемые типы данных как локальные и глобальные переменные  
Если переменная объявлена внутри функции - она локальная
```python
# Изменяемые объекты:
list, dict, set, bytearray

# Неизменяемые объекты:
int, float, str, tuple, complex, frozenset
```
```python
# int - неизменяемый
x = 10
def foo(x):
    x += 5 # создается новая локальная переменная
    print(x)

foo(x) # Выводит 15
print(x) # Выводит 10
```
```python
# list - изменяемый
l = [10]
def bar(l):
    l.append(15) # изменяется глобальная переменная
    print(l)

bar(l) # Выводит [10, 15]
print(l) # Выводит [10, 15]
```

Полезные ссылки  
[cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/opinions/)
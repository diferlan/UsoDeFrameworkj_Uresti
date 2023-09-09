# %%
import pandas as pd #Con pandas se hace el análisis, manipulación de datos y leer el archivo que contiene los datos
from sklearn.model_selection import train_test_split #Con train_test_split es una función que se usa para dividir los datos en 2 conjuntos, de entrenamiento y de prueba
#La libreria de metrics de sklearn incluye variedad de funciones para evaluar nuestro modelo
#Importamos funciones para calcular el accuracy, precision, f1 y recall. Además importamos una función para armar la matriz de confusión e identificar clasficaciones correctas, falsos negativos y falsos negativos
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler  #De la libreria de preprocesamiento de sklearn importamos un escalados que transforma nuestras caracteristicas en un rango de 0 a 1, nos ayuda a evita problemas de escala en las diferentes variables
from sklearn.ensemble import GradientBoostingClassifier #Esta función implementa el algoritmo nombrado, es usado para problemas de clasificación, una de sus ventajas es que combina  varios modelos para crear un modleo robusto
import time #Esta biblioteca incluye funcione spara medir el tiempo en python
import matplotlib.pyplot as plt #Esta es una de las principales bibliotecas de visualización en Python, se usa para crear gráficos y en este caso explicar los resultados
from sklearn.model_selection import cross_val_score, KFold
from mlxtend.evaluate import bias_variance_decomp

#%%
print('Inicializando algoritmo de Gradient Boosting Classifier...') #Imprime un mensaje para que el usuario sepa que esta iniciando el programa
time.sleep(1) #Usamos sleep para espaciar ligeramente los print y se puedan leer
data = pd.read_csv('stroke_data.csv') #Usamos la función read:csv de pandas para leer el dataset que se usará
print('Asegurese que es encuentra en la misma carpeta que el archivo .csv') #Pequeño mensaje, la idea es que se lea si el programa da error por no encontrar el archivo
time.sleep(1) #Esperamos un segundo para que se puedan leer los mensajes
data.dropna(subset=['sex'],inplace=True) #Eliminamos los datos faltantes para evitar problemas
# División de datos en conjuntos de entrenamiento y prueba
x = data.drop('stroke', axis=1)  #Elimina la variables a predecir 'stroke' del conjunto x que contiene los predictores
y = data['stroke'] #Selecciona la variable objetivo
#Se escalan las datos, esto para evitar que los resultados se distorcionen
escalar = MinMaxScaler() #Creación del objeto de escalamiento, usamos la función MinMaxScaler, todos los datos tomaran valores entre 0 y 1
X = escalar.fit_transform(x) #escalamiento de los predictores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6) #División de los datos en 2 conjuntosm entrenamiento y prueba
#Imprimimos un pequeño mensaje para que el usuario sepa que se esta haciendo algo y no se trabo el programa
print('Entrenando...')
time.sleep(2)
#Crea un modelo de Gradient Boosting Classifier, usamos un learning rate bajo y suficientes estimadores para que el modelo se ajuste de la mejor manera, intentando evitar overfitting
gbc = GradientBoostingClassifier(learning_rate=0.05,n_estimators=2000)

# Entrena el modelo en los datos de entrenamiento
gbc.fit(X_train, y_train)
# Realiza las predicciones en el conjunto de prueba usando el modelo
y_pred = gbc.predict(X_test)
#Extraemos los resultados con diferentes métricas y los imprimos, usamos una precisión a 4 decimales para notar las pequeñas diferencias
accuracy,recall,precision,f1 = accuracy_score(y_test, y_pred), recall_score(y_test,y_pred) , precision_score(y_test,y_pred), f1_score(y_test,y_pred)
print(f'Accuracy en el conjunto de prueba: {accuracy:.4f}')
print(f'Recall en el conjunto de prueba: {recall:.4f}')
print(f'Precision en el conjunto de prueba: {precision:.4f}')
print(f'F1 en el conjunto de prueba: {f1:.4f}')
#Imprimimos la matriz de confusión para ver en que esta fallando el modelo
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
time.sleep(4)
#Extraemos la importancia de cada predictor para ver variables el modelo considera más importantes e influyen más en la clasificación
feature_importances = gbc.feature_importances_
'''
plt.figure(figsize=(10, 8)) #Inicializamos una figura con un tamaño de 10largox8alto donde se pondrá la gráfica
#Creamos un gráfico de barras horizontales, cada barra representa un predictor y los nombres en y corresponden al nombre del predictor que describe la barra
plt.barh(range(len(feature_importances)), feature_importances, tick_label=x.columns)
plt.xlabel('Características') #Pone el nombre del eje X
plt.ylabel('Importancia') #Pone el nombre del eje Y
plt.title('Importancia de las características en el modelo Gradient Boosting Classifier') #Incluye un titulo en la grafica
plt.show() #Muestra la figura
'''
# %%
print('Realizando Cross Validation...')
folds = KFold(n_splits=3,shuffle=True)
score = cross_val_score(gbc,X,y,cv=folds)
print("Fold scores: ",score)
print("Mean Cross Validation score: ",score.mean())
test_ = folds.split(X,y)
print(test_)
folds_index=[]
for i, (train_index, test_index) in enumerate(folds.split(X)):
    print(f"Fold {i}:")
    print(f"  Test:  index={test_index}")
    folds_index.append(test_index)
colorlist=[]
foldno=[]
for j in range(3):
    colors = ['r' if str(i) in str(folds_index[j]) else 'g' for i in range(len(colors))]
    colorlist.append(colors)
    temp = [j] * X.shape[0]
    foldno.append()

for i in range(3):
    plt.scatter(foldno[i],range(len(X)),c=colorlist[i])

    


plt.bar(range(1,4),score)
plt.xlabel('# de fold')
plt.ylabel('Score')

_, bias, var = bias_variance_decomp(gbc, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)
#print('R^2: %.4f' % loss)
print('Bias: %.4f' % bias)
print('Variance: %.4f' % var)

"""
color = arreglo de n puntos
color = modificar para que puntos en test tengan otro color
plt.scatter(folds split,colors)
"""
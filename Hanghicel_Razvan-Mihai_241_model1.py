"""
Modelul prezentat este SVM
Mai jos am importat modulele de care o sa am nevoie pentru citirea imaginilor, procesarea lor, definirea modelului,
antrenarea acestuia si afisarea scorurilor si a matricei de confuzie.
"""

import numpy as np
import os
from sklearn import svm
from skimage.io import imread
import matplotlib.pyplot as plt
import csv
import pandas
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""
    Acum impart datele din fisierul train_labels.txt in 2 liste: o lista care va retine numele imaginii sub forma de string si alta lista unde se vor retine label-urile
imaginilor. Pentru a citi datele din train_labels.txt am folosit pandas care returneaza un DataFrame cu numele imaginilor si label-urile. Apoi in train_images_data o sa 
iau datele din coloana "id" din DataFrame si le convertesc intr-o lista si la fel si pentru train_images_labels care va lua datele din coloana "class" si le transforma intr-o
lista. 
"""
train_images_data = []
train_images_labels = []

data = pandas.read_csv('./train_labels.txt')
train_images_data = data["id"].tolist()
train_images_labels = data["class"].tolist()
"""
    Atunci cand transform datele din coloana "id" intr-o lista ele vor fi convertite la int, doar ca pentru a usura procesarea datelor o sa convertesc la string si pentru
fiecare numar o sa-i adaug in fata zerouri astfel incat lungimea string-ului sa fie 6.
"""
for i in range(len(train_images_data)):
    if train_images_data[i] < 10:
        train_images_data[i] = "00000" + str(train_images_data[i])
    elif train_images_data[i] < 100:
        train_images_data[i] = "0000" + str(train_images_data[i])
    elif train_images_data[i] < 1000:
        train_images_data[i] = "000" + str(train_images_data[i])
    elif train_images_data[i] < 10000:
        train_images_data[i] = "00" + str(train_images_data[i])
    elif train_images_data[i] < 100000:
        train_images_data[i] = "0" + str(train_images_data[i])
    else:
        train_images_data[i] = str(train_images_data[i])

"""
    Pentru datele de validare fac aceeasi procesare ca cea descrisa mai sus.
"""

validation_images_data = []
validation_images_labels = []
data = pandas.read_csv('./validation_labels.txt')
validation_images_data = data["id"].tolist()
validation_images_labels = data["class"].tolist()

for i in range(len(validation_images_data)):
    if validation_images_data[i] < 10:
        validation_images_data[i] = "00000" + str(validation_images_data[i])
    elif validation_images_data[i] < 100:
        validation_images_data[i] = "0000" + str(validation_images_data[i])
    elif validation_images_data[i] < 1000:
        validation_images_data[i] = "000" + str(validation_images_data[i])
    elif validation_images_data[i] < 10000:
        validation_images_data[i] = "00" + str(validation_images_data[i])
    elif validation_images_data[i] < 100000:
        validation_images_data[i] = "0" + str(validation_images_data[i])
    else:
        validation_images_data[i] = str(validation_images_data[i])

"""
    Acum pentru datele de test o sa salvez in image_files toate imaginile din folder-ul "data" care se termina cu ".png". Apoi aplicam procesarea 
"""
image_dir = "data/"
image_files = np.array([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")])

test_images_data = []
for i in range(17001, len(image_files) + 1):
    if i < 100000:
        test_images_data.append("0" + str(i))
    else:
        test_images_data.append(str(i))

"""
    Acum pentru datele de test, train si validation o sa citesc imaginile folosind cv2 si la care aplic cv2.IMREAD_GRAYSCALE care va transforma shape-ul imaginilor din
(224,224,3) in (224,224). Apoi pentru fiecare linie din imagine o sa calculez media pe acea linie pentru a fi siguri ca imaginile o sa incapa in RAM si pentru a scadea
timpul de executie al programului.
"""
reshaped_test_data = []
for current_id in test_images_data:
    image = cv2.imread('data/' + current_id + '.png',cv2.IMREAD_GRAYSCALE)
    image_data = []
    for j in image:
        image_data.append(j.mean())
    reshaped_test_data.append(image_data)

reshaped_train_data = []
for current_id in train_images_data:
    image = cv2.imread('data/' + current_id + '.png',cv2.IMREAD_GRAYSCALE)
    image_data = []
    for j in image:
        image_data.append(j.mean())
    reshaped_train_data.append(image_data)

reshaped_validation_data = []
for current_id in validation_images_data:
    image = cv2.imread('data/' + current_id + '.png',cv2.IMREAD_GRAYSCALE)
    image_data = []
    for j in image:
        image_data.append(j.mean())
    reshaped_validation_data.append(image_data)

"""
    Acum o sa calculez numarul de aparitii pentru 1 si 0 in label-urile de train. Pentru fiecare dintre cele doua label-uri o sa asignez un weight, deoarece avem o
diferenta destul de mare intre numarul de 0 fata de 1 si sa balansez datele.
"""
one_weight = train_images_labels.count(1)
one_weight = 1/one_weight
zero_weight = train_images_labels.count(0)
zero_weight = 1/zero_weight

"""
    Definesc modelul SVM la care adaug class_weight unde o sa pun un dictionar cu cheia numele fiecarui label si ca valoare va fi weight-ul calculat anterior.
"""
svm_model = svm.SVC(C = 1, kernel="poly", class_weight={1: one_weight, 0: zero_weight})
"""
    Dau fit modelului cu datele de train si cu label-urile de train.
"""
svm_model.fit(reshaped_train_data, train_images_labels)
"""
    Acum o sa prezicem label-urile datelor de validare pe care le stochez in predicted_validation.
"""
predicted_validation = svm_model.predict(reshaped_validation_data)
"""
    Se calculeaza acuratetea pe datele de validare, adica verific ce procent din label-urile prezise mai sus sunt egale cu label-urile actuale.
"""
acc_validation = accuracy_score(validation_images_labels, predicted_validation)
"""
    Afisez acuratetea obtinuta anterior
"""
print("Accuracy validation: ", acc_validation)
"""
    Afisez precission score
"""
print("Precission score: ", precision_score(validation_images_labels, predicted_validation))
"""
    Afisez recall score
"""
print("Recall score: ", recall_score(validation_images_labels, predicted_validation))
"""
    Afisez F1 score
"""
print("F1 score: ", f1_score(validation_images_labels, predicted_validation))
"""
    Acum o sa prezicem label-urile pentru datele de test
"""
predicted_test = svm_model.predict(reshaped_test_data)
"""
    Acum urmeaza partea de afisare a label-urilor de test prezise mai sus in fisierul "sample.csv". O sa deschid fisierul cu drepturi de write si adaug initial prima linie
cu "id,class". Apoi iterez prin lista de label-uri si adaug numele imaginii urmat de label-ul prezis si la final inchid fisierul.
"""
f = open("sample.csv", "w")
f.write("id,class\n")
for i in range(len(predicted_test)):
    f.write(str(test_images_data[i])+","+str(predicted_test[i])+"\n")
f.close()

"""
    Construiesc matrice de confuzie pentru datele de validare
"""
confusion_matrix_SVM = confusion_matrix(y_true=validation_images_labels,
                                       y_pred=predicted_validation)
"""
    Acum pentru a arata vizual matricea de confuzie o sa folosesc subplots din matplotlib. Apoi folosesc matshow pentru axes care va crea o matrice de plotare care va prelua
matricea de confuzie si ca colormap va fi cm.Purples. Acum o sa iterez cu 2 for-uri prin valorile din matricea de confuzie si o sa adaug valoarea de la pozitia curenta in celula
determinata de x si y si o sa adaug ca va (vertical alignment) sa fie "center" si la fel si pentru ha (horizontal alignment) sa fie "center", asfel valoarea se va afisa in centrul 
celulei. La final setez label-ul pentru axa x ca "Label-uri prezise" si pentru axa y ca "Label-uri initiale" si afisez reprezentarea.
"""
figure, axes = plt.subplots()
axes.matshow(confusion_matrix_SVM, cmap=plt.cm.Purples)
for i in range(confusion_matrix_SVM.shape[0]):
    for j in range(confusion_matrix_SVM.shape[1]):
        axes.text(x=j,y=i,s=confusion_matrix_SVM[i,j],va='center',ha='center')
plt.xlabel('Label-uri prezise')
plt.ylabel('Label-uri initiale')
plt.title('SVM')
plt.show()
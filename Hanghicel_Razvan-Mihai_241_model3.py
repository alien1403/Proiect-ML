"""
Modelul prezentat este Convolutional Neural Network
Mai jos am importat modulele de care o sa am nevoie pentru citirea imaginilor, procesarea lor, definirea modelului,
antrenarea acestuia si afisarea scorurilor si a matricei de confuzie.
"""
import pandas as pd
import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""
    Verific daca modelul poate fi antrenat pe GPU sau pe CPU
"""
device = ""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

"""
    Definesc un dataFrame care va retine numele_imaginii si label-ul corespunzator pe care le vom folosi mai tarziu cand o sa definesc propiul dataset. Apoi cu pandas am citit datele din fisierul
de train_labels si in train_images_data retin numele imaginilor intr-o lista sub forma de int-uri pe care o sa le transform in string-uri prin adaugarea de 0-uri astfel incat sa ajung cu lungimile la 6
pentru toate numele imaginilor. Apoi train_images_labels va retine o lista cu label-urile imaginilor. Asignez valorile celor doua liste catre dataFrame-ul definit anterior si la final salvez rezultatul
in train_csv.csv in care scot indexul pentru fiecare linie si pastrez header-ul.
"""
train_dataFrame = pd.DataFrame(columns = ["img_name","label"])

data = pd.read_csv('/kaggle/input/unibuc-brain-ad/data/train_labels.txt')
train_images_data = data["id"].tolist()
train_images_labels = data["class"].tolist()

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
train_dataFrame["img_name"] = train_images_data
train_dataFrame["label"] = train_images_labels

train_dataFrame.to_csv(r'train_csv.csv', index = False, header = True)


"""
    Pentru datele de validare am facut exact acelasi lucru ca la datele de train explicat mai sus.
"""

validation_dataFrame = pd.DataFrame(columns = ["img_name", "label"])

data = pd.read_csv('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt')
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
        
validation_dataFrame["img_name"] = validation_images_data
validation_dataFrame["label"] = validation_images_labels

validation_dataFrame.to_csv(r'validation_csv.csv', index = False, header = True)


"""
    Pentru datele de test am facut exact acelasi lucru ca la datele de train explicat mai sus, doar ca test_images_labels va avea toate valorile de 0 implicit, adica am setat initial ca toate
label-urile de la setul de test sa fie 0.
"""

test_dataFrame = pd.DataFrame(columns = ["img_name", "label"])
image_dir = "/kaggle/input/unibuc-brain-ad/data/data/"
image_files = np.array([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")])

test_images_data = []

test_images_labels = []
for i in range(17001, len(image_files) + 1):
    if i < 100000:
        test_images_data.append("0" + str(i))
    else:
        test_images_data.append(str(i))
    test_images_labels.append(0)
        
test_dataFrame["img_name"] = test_images_data
test_dataFrame["label"] = test_images_labels

test_dataFrame.to_csv(r'test_csv.csv', index = False, header = True)



"""
    Acum definesc propriul Dataset pe care o sa-l folosesc mai departe pentru a putea procesa imaginile. In constructor am definit o variabile pentru path-ul unde se afla imaginile,
fisierul unde se afla label-urile si transformarea daca este cazul. Apoi am definit functia de __len__ care returneaza lungimea listei de labels si __getitem__ care in functie de un index
va returna imaginea si label-ul corespunzator acesteia
"""

class CustomDataset(Dataset):
    def __init__(self, root_dir, labels, transform):
        self.root_dir = root_dir
        self.labels = pd.read_csv(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        img_id = self.annotations.iloc[index, 0] # preia numele imaginii cu indexul index, dar pe care il transforma in int
        # mai jos am transformarea acelui int in string prin adaugarea de 0-uri in fata pana ajunge la lungimea de 6 caractere
        if img_id < 10:
            img_id = "00000" + str(img_id)
        elif img_id < 100:
            img_id = "0000" + str(img_id)
        elif img_id < 1000:
            img_id = "000" + str(img_id)
        elif img_id < 10000:
            img_id = "00" + str(img_id)
        elif img_id < 100000:
            img_id = "0" + str(img_id)
        else:
            img_id = str(img_id)
        # deschid imaginea de la ruta unde se afla imaginea si cu numele imaginii la care concatenez ".png" pentru a adauga extensia acesteia
        img = Image.open(os.path.join(str(self.root_dir),str(img_id) + ".png"))
        label = torch.tensor(int(self.labels.iloc[index, 1])) # preia label-ul imaginii cu indexul index
        # daca transform nu este specificat, adica nu aplicam nicio transformare imaginii o sa returnam imaginea si cu label-ul corespunzator, altfel o sa aplicam transformarea imaginii si apoi returnam
        if self.transform is not None:
            img = self.transform(img)
        return (img, label)

"""
    Definesc ce data processing si data augmentation o sa aplic fiecarei imagini. In acest caz aplic rotirea pe verticala, rotirea pe orizontala, 
    rotirea in jurul centrului imaginii, normalizarea in care o sa se scada media si o sa se imparta la deviatia standard pentru fiecare channel si la final transform in tensor.
"""


image_transform = transforms.Compose([transforms.RandomVerticalFlip(),  transforms.RandomHorizontalFlip(), transforms.RandomRotation(80), transforms.ToTensor(), transforms.Normalize(
                                        mean = [0.5, 0.5, 0.5],
                                        std = [0.5, 0.5, 0.5])
                                     ])
batch_size = 64
num_classes = 2
learning_rate = 0.0001
num_epochs = 100
num_workers = 2

"""
    Am impartit datele in 3 foldere: train, validation si test, unde am adaugat imaginile respective pentru fiecare categorie din cele 3. Apoi pentru fiecare categorie definesc un dataset
in care incarc datele din folder-ul respectiv, cu csv-ul creat anterior si aplic data augmentation-ul definit anterior
"""

dataset_train = CustomDataset("/kaggle/input/mydataset/train/", "/kaggle/working/train_csv.csv", transform=image_transform)
dataset_validation = CustomDataset("/kaggle/input/mydataset/validation/", "/kaggle/working/validation_csv.csv", transform = image_transform)
dataset_test = CustomDataset("/kaggle/input/mydataset/test/", "/kaggle/working/test_csv.csv", transform=image_transform)
"""
    Definesc pentru fiecare categorie cate un DataLoader in care adaug ca dataset fiecare dataset definit anterior, pentru datele de train o sa fie shuffle = True, iar pentru celelalte doua 
shuffle = False. Apoi batch_size-ul va fi cel definit mai sus si setez si num_workers cu valoarea de mai sus
"""
train_loader = DataLoader(dataset=dataset_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
validation_loader = DataLoader(dataset=dataset_validation, shuffle=False, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(dataset=dataset_test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

"""
    Definesc clasa ConvolutionalNeuralNetwork in care o sa creez modelul propriu-zis. Constructorul va avea ca parametru numarul de clase pentru care trebuie sa clasificam imaginile.
Primul layer va primi ca input 3 canale si va returna 64 de canale cu un kernel de 3x3, stride de 1x1 si padding de 1x1. Al doilea layer va primi ca input 64 canale si va returna 128 de canale
cu un kernel de 3x3, stride de 1x1 si padding de 1x1. Al treilea layer va primi ca input 128 canale si va returna 256 de canale cu un kernel de 3x3, stride de 1x1 si padding de 1x1.
Al patrulea layer va primi ca input 256 canale si va returna 512 de canale cu un kernel de 3x3, stride de 1x1 si padding de 1x1. Max_pool care aplica MaxPool2d cu kernel de 2x2 si stride de 2x2.
In final avem 5 fully connected layers de tip linear. Apoi in metoda forward se arata efectiv cum plecand de la inputul x o sa ajungem la output-ul dorit.

"""

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3),stride = (1,1), padding = (1,1))
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.conv_layer3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        self.conv_layer4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride = (1,1), padding = (1,1))
        
        self.max_pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.full_con1 = nn.Linear(100352, 100352//128)
        self.full_con2 = nn.Linear(100352//128, 100352//256)
        self.full_con3 = nn.Linear(100352//256, 100352//512)
        self.full_con4 = nn.Linear(100352//512, 100352//1024)
        self.full_con5 = nn.Linear(100352//1024, num_classes)
       
    def forward(self, x):
        out = self.conv_layer1(x)
        out = F.relu(out)
        out = self.max_pool(out)
        
        out = self.conv_layer2(out)
        out = F.relu(out)
        out = self.max_pool(out)
        
        out = self.conv_layer3(out)
        out = F.relu(out)
        out = self.max_pool(out)
        
        out = self.conv_layer4(out)
        out = F.relu(out)
        out = self.max_pool(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = F.relu(self.full_con1(out))
        out = F.relu(self.full_con2(out))
        out = F.relu(self.full_con3(out))
        out = F.relu(self.full_con4(out))
        out = self.full_con5(out)
        return out

"""
    Definesc modelul si verific daca cuda este instalata si in caz pozitiv o sa pun modelul pe GPU pentru a se antrena
"""

model = ConvolutionalNeuralNetwork(num_classes)
if torch.cuda.is_available():
    model.cuda()

"""
    Pentru functia de loss o sa folosesc CrossEntropyLoss 
"""
crt = nn.CrossEntropyLoss()

"""
    Pentru optimizator folosesc Adam care va optimiza parametrii modelului cu learning rate-ul definit mai sus
"""

optimizer = torch.optim.Adam(model.parameters(),
                            lr=learning_rate)  

"""
    Acum iterez printr-un numar de epoci definit mai sus si prin datele de training si salvez pentru fiecare epoca ce loss pentru a putea reprezenta grafic cum se modifica loss-ul.
"""
loss_values = []
import math
for epoch in range(num_epochs):
    print(epoch)
    for i, (images, labels) in enumerate(train_loader):  
        if i % 20 == 0:
            print("\tProgress:{}".format(math.floor(i/len(train_loader) * 100)))
            
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = crt(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item())) 
    loss_values.append(loss.item())
       

#
"""
    Adaug imaginile de validation pentru a prezice label-urile si la final afisez acuratetea cu care a determinat modelul label-urile.
"""

predicted_validation_labels = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_validation_labels.append(predicted.tolist())
    
    print('Accuracy of the network on the {} validation images: {} %'.format(2000, 100 * correct / total))


"""
    Dupa codul de mai sus o sa obtinem o matrice si mai jos o sa transform matricea in lista.
"""


predicted_validation = []

for i in range(len(predicted_validation_labels)):
    for j in range(len(predicted_validation_labels[i])):
        predicted_validation.append(predicted_validation_labels[i][j])

acc_validation = accuracy_score(validation_images_labels, predicted_validation)
"""
    Afisez acuratetea obtinuta anterior
"""
print("Accuracy validation: ", acc_validation)
"""
    Afisez precission score
"""
print("Precission score: ",precision_score(validation_images_labels, predicted_validation))
"""
    Afisez recall score
"""
print("Recall score: ",recall_score(validation_images_labels, predicted_validation))
"""
    Afisez F1 score
"""
print("F1 score: ",f1_score(validation_images_labels, predicted_validation))
"""
    Acum o sa prezicem label-urile pentru datele de test
"""
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            predictions.append(predicted[i])
        total += len(predicted)

predicted_test = []
for i in range(len(predictions)):
    predicted_test.append(predictions[i].tolist())

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
    Definirea matricei de confuzie
"""
confusion_matrix_CNN = confusion_matrix(y_true=validation_images_labels, y_pred=predicted_validation)

"""
    Acum pentru a arata vizual matricea de confuzie o sa folosesc subplots din matplotlib. Apoi folosesc matshow pentru axes care va crea o matrice de plotare care va prelua
matricea de confuzie si ca colormap va fi cm.Purples. Acum o sa iterez cu 2 for-uri prin valorile din matricea de confuzie si o sa adaug valoarea de la pozitia curenta in celula
determinata de x si y si o sa adaug ca va (vertical alignment) sa fie "center" si la fel si pentru ha (horizontal alignment) sa fie "center", asfel valoarea se va afisa in centrul 
celulei. La final setez label-ul pentru axa x ca "Label-uri prezise" si pentru axa y ca "Label-uri initiale" si afisez reprezentarea.
"""

confusion_matrix_CNN = confusion_matrix(validation_images_labels,predicted_validation)
figure, axes = plt.subplots()
axes.matshow(confusion_matrix_CNN, cmap=plt.cm.Purples)
for i in range(confusion_matrix_CNN.shape[0]):
    for j in range(confusion_matrix_CNN.shape[1]):
        axes.text(x=j,y=i,s=confusion_matrix_CNN[i,j],va='center',ha='center')
plt.xlabel('Label-uri prezise')
plt.ylabel('Label-uri initiale')
plt.title('Convolutional Neural Network')
plt.show()

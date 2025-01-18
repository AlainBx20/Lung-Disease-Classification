from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib
import numpy as np
import os 

def load_image(folder,label_v,target_size=(64,64)):
    data=[]
    label=[]
    for file in os.listdir(folder):
        img_p=os.path.join(folder,file)
        img=imread(img_p)
        img_normalize=resize(img,target_size,anti_aliasing=True)
        data.append(img_normalize.flatten())
        label.append(label_v)
    return np.array(data), np.array(label)

normal_d, normal_l = load_image('Normal', 0)
opacity_d, opacity_l = load_image('Lung_Opacity', 1)


X = np.concatenate([normal_d, opacity_d])
y = np.concatenate([normal_l, opacity_l])

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=SVC(kernel='linear',probability=True)
model.fit(x_train,y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)


joblib.dump(svm_model, 'lung_disease_model.pkl')



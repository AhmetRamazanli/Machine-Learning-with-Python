#*********************************KÜTÜPHANELER********************************
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#********************************VERİ YÜKLEME*********************************
dosya = pd.read_excel('StudentsPerformance.xlsx') #dosya oku 


sayisalveriler = dosya.iloc[:,5:8].values
print(sayisalveriler)

#*************************EKSİK VERİLERİN DÜZENLENMESİ************************
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean' ,verbose=0)

imputer = imputer.fit(sayisalveriler[:, 0:3])   #burada 5:8 yazdığımda hata alıyorum
sayisalveriler[:, 0:3] = imputer.transform(sayisalveriler[:, 0:3])

#******************KATEGORİK VERİLERİN SAYISALLAŞTIRILMASI********************
#*******_____0_____*******
gender = dosya.iloc[:,0:1].values
le = LabelEncoder()
ohe = ColumnTransformer([("gender", OneHotEncoder(), [0])], remainder = 'passthrough')
gender[:,0] = le.fit_transform(gender[:,0])
print(gender)
gender = ohe.fit_transform(gender)
print(gender)

#*******_____1_____*******
race = dosya.iloc[:,1:2].values
le1 = LabelEncoder()
ohe1 = ColumnTransformer([("race/ethnicity", OneHotEncoder(), [0])], remainder = 'passthrough')
race[:,0] = le1.fit_transform(race[:,0])
print(race)
race = ohe1.fit_transform(race).toarray()
print(race)

#*******_____2_____*******
parentaldegree = dosya.iloc[:,2:3].values
le2 = LabelEncoder()
ohe2 = ColumnTransformer([("ple", OneHotEncoder(), [0])], remainder = 'passthrough')
parentaldegree[:,0] = le2.fit_transform(parentaldegree[:,0])
print(parentaldegree)
parentaldegree = ohe2.fit_transform(parentaldegree).toarray()
print(parentaldegree)

#*******_____3_____*******
lunch = dosya.iloc[:,3:4].values
le3 = LabelEncoder()
ohe3 = ColumnTransformer([("lunch", OneHotEncoder(), [0])], remainder = 'passthrough')
lunch[:,0] = le3.fit_transform(lunch[:,0])
print(lunch)
lunch = ohe3.fit_transform(lunch)
print(lunch)

#*******_____4_____*******
test_prep_course = dosya.iloc[:,4:5].values
le4 = LabelEncoder()
ohe4 = ColumnTransformer([("test_prep_course", OneHotEncoder(), [0])], remainder = 'passthrough')
test_prep_course[:,0] = le4.fit_transform(test_prep_course[:,0])
print(test_prep_course)
test_prep_course = ohe4.fit_transform(test_prep_course)
print(test_prep_course)

#***********************VERİLERİN BİRLEŞTİRİLMESİ*****************************
print(list(range(1000)))

sonuc = pd.DataFrame(data = gender, index = range(1000), columns=['female','male'] )
print(sonuc)

sonuc1 = pd.DataFrame(data = race, index = range(1000), columns=['group A','group B','group C', 'group D','group E'] )
print(sonuc1)

sonuc2 = pd.DataFrame(data = parentaldegree, index = range(1000), columns=['associate degree','bachelor degree','high school','master degree','some college','some high school'] )
print(sonuc2)

sonuc3 = pd.DataFrame(data = lunch, index = range(1000), columns=['free/reduced','standard'] )
print(sonuc3)

sonuc4 = pd.DataFrame(data = test_prep_course, index = range(1000), columns=['completed','none'] )
print(sonuc4)

sonuc5 = pd.DataFrame(data = sayisalveriler, index = range(1000), columns=['Math Score','Reading Score','Writing Score'] )
print(sonuc5)

s=pd.concat([sonuc2,sonuc3],axis=1)
print(s)

s1=pd.concat([s,sonuc4],axis=1)
print(s1)

s2=pd.concat([s1,sonuc5],axis=1)
print(s2)

s3=pd.concat([s2,sonuc1],axis=1)
print(s3)

s4=pd.concat([s3,sonuc],axis=1)
print(s4)

#************************EĞİTİM VE TESTLERE BÖLME*****************************
x_train,x_test, y_train , y_test= train_test_split(s2,sonuc1,test_size=0.33,random_state=0)

x_train1,x_test1, y_train1 , y_test1= train_test_split(s3,sonuc,test_size=0.33,random_state=0)

#*************************ÖZNİTELİK ÖLÇEKLENDİRME*****************************
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_train)

sc1=StandardScaler()
x_train1=sc1.fit_transform(x_train1)
x_test1=sc1.fit_transform(x_test1)
print(x_train1)

#*************************DOĞRUSAL_REGRESYON_TAHMİN***************************
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)

lr1=LinearRegression()
lr1.fit(x_train1,y_train1)

tahmin1=lr1.predict(x_test1)

#**************************POLİNOM_REGRESYON_TAHMİN***************************
#poly_reg = PolynomialFeatures(degree = 2)
#polinom dönüşümü 2.dereceden 
#x_poly = poly_reg.fit_transform(X)
#print(x_poly)
#lin_reg2 = LinearRegression()
#lin_reg2.fit(x_poly,y)

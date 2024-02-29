#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[3]:


genshin_cstats = pd.read_csv("genshinstats_1.csv")

Stats_90 = genshin_cstats[(genshin_cstats['Lv'] == 90) ]
# Main Role by Element
plt.figure(figsize=(8,6))
sns.countplot(x='Weapon', data=Stats_90, hue='Element', palette='pastel')
plt.title("Weapon of Character by Element Type")

plt.legend(loc='best')

plt.show()


# ### เป็นกราฟแสดงตัวละครที่ใช้อาวุธต่างๆโดยจำแนกตามธาตุของตัวละคร 
# ### โดยที่แกน X จะแสดงประเภทของอาวุธ และแกน Y จะแสดงจำนวนของตัวละคร
# 

# In[4]:


df1 = pd.read_csv('genshinstats_1.csv')

x = df1[['Base ATK', 'Base HP','Base DEF']]
y = df1['Main role']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

sns.scatterplot(data=df1, x='Base ATK', y='Base HP', hue='Main role', size='Base DEF')

new_character = [[340,13372,548]]
print(model.predict(new_character))
plt.scatter(x = [340], y = [13372] , sizes = [60],c = 'pink')
model.score(x_test, y_test)
print(model.score(x_test, y_test))
plt.show()


# #### เป็น Model ที่ใช้สำหรับทำนายตำแหน่งของตัวละครในเกม Genshin Impact โดยคำนวณจาก Status พื้นฐานของตัวละคร ซึ่งเกม Genshin Impact ข้อมูลของตัวละครที่ออกใหม่จะบอกเพียง Status และ Skill ของตัวละครเท่านั้น ทำให้เวลาที่ตัวละคร ใหม่ออกมา ทำให้ผู้เล่นจำเป็นต้องคาดการณ์วิธีใช้ตัวละครนั้นด้วยตัวเอง ซึ่งในบางครั้งตัวละครที่ถูกปล่อยออกมาใหม่นั้นไม่สามารถเล่นในตำแหน่งอย่างที่คาดการเอาไว้ ทำให้ผู้เล่นผิดหวังจนเกิดดราม่าอยู่บ่อยครั้ง ซึ่งตัวของ Model นีั้ก็สามารถช่วยในการคาดเดาเบื้องต้นให้กับผู้เล่นได้

# In[ ]:





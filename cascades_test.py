#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


smileCascade = cv2.CascadeClassifier('C:\\Users\\minja\\OneDrive\\Documents\\haarcascade_smile.xml')
faceCascade = cv2.CascadeClassifier('C:\\Users\\minja\\OneDrive\\Documents\\haarcascade_frontalface_default.xml')


# In[14]:


cap = cv2.VideoCapture(0)
smilesDetected = 0
noSmiles = 2
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.2,5)


    
    for (xf, yf, wf, hf) in faces:
        cv2.rectangle(gray, (xf,yf),(xf+wf, yf+hf),(200,200,100),4)
        cv2.rectangle(img, (xf,yf),(xf+wf, yf+hf),(200,200,100),4)
        faceROI = gray[yf:yf+hf,xf:xf+wf]
        faceROIcolor = img[yf:yf+hf,xf:xf+wf]
        smiles = smileCascade.detectMultiScale(faceROI, 1.8,10)
        for (x,y,w,h) in smiles:
            
            cv2.rectangle(faceROIcolor,(x,y),(x+w,y+h), (200,200,200), 2)
            cv2.rectangle(faceROI,(x,y),(x+w,y+h), (200,200,200), 2)
    
    if len(smiles)>0:
        smilesDetected+=1
        noSmiles=0

    else:
        noSmiles+=1
        smilesDetected=0
            
        
    if smilesDetected>1:
        imToShow = img    
    elif noSmiles>1:
        imToShow = gray

        
        
    
            
    cv2.imshow('img',imToShow)
    k = cv2.waitKey(30) & 0xff
    
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
        


# In[ ]:





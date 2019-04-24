#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import cv2


# In[39]:


smileCascade = cv2.CascadeClassifier('C:\\Users\\minja\\OneDrive\\Documents\\haarcascade_smile.xml')
faceCascade = cv2.CascadeClassifier('C:\\Users\\minja\\OneDrive\\Documents\\haarcascade_frontalface_default.xml')


# In[61]:


cap = cv2.VideoCapture(0)


# In[62]:


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.5,3)
    smiles = smileCascade.detectMultiScale(gray, 1.6,5)
    
    for (xf, yf, wf, hf) in faces:
        cv2.rectangle(img, (xf,yf),(xf+wf, yf+hf),(0,200,100),2)
        faceROI = gray[yf:yf+hf,xf:xf+wf]
        faceROIcolor = img[yf:yf+hf,xf:xf+wf]
        smiles = smileCascade.detectMultiScale(faceROI, 1.6,5)
        for (x,y,w,h) in smiles:
            
            cv2.rectangle(faceROIcolor,(x,y),(x+w,y+h), (255,0,0), 2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
        


# In[ ]:





# In[ ]:





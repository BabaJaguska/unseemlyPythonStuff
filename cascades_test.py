#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import cv2


# In[26]:


smileCascade = cv2.CascadeClassifier('C:\\Users\\minja\\OneDrive\\Documents\\haarcascade_smile.xml')
faceCascade = cv2.CascadeClassifier('C:\\Users\\minja\\OneDrive\\Documents\\haarcascade_frontalface_default.xml')


# In[64]:


import numpy as np


# In[79]:


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3)) # it will not accept float!!!
frame_height = int(cap.get(4))

out = cv2.VideoWriter('recording.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, (frame_width,frame_height))

smilesDetected = 0
noSmiles = 2

while True:
    ret, img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')
    h,s,v = cv2.split(hsv)
    adjustment = 0.1
    s = s*adjustment
    s = np.clip(s,0,255)
    desatHSV = cv2.merge([h,s,v])
    desatBGR = cv2.cvtColor(desatHSV.astype("uint8"),cv2.COLOR_HSV2BGR)
    
    faces = faceCascade.detectMultiScale(gray,1.2,5)
    
    for (xf, yf, wf, hf) in faces:
        cv2.rectangle(desatBGR, (xf,yf),(xf+wf, yf+hf),(200,200,100),4)
        cv2.rectangle(img, (xf,yf),(xf+wf, yf+hf),(200,200,100),4)
        faceROI = gray[yf:yf+hf,xf:xf+wf]
        faceROIcolor = img[yf:yf+hf,xf:xf+wf]
        faceROIdesat = desatBGR[yf:yf+hf,xf:xf+wf]
        smiles = smileCascade.detectMultiScale(faceROI, 1.8,10)
        for (x,y,w,h) in smiles:
            
            cv2.rectangle(faceROIcolor,(x,y),(x+w,y+h), (200,200,200), 2)
            cv2.rectangle(faceROIdesat,(x,y),(x+w,y+h), (200,200,200), 2)
            
    
    if len(smiles)>0:
        smilesDetected+=1
        noSmiles=0

    else:
        noSmiles+=1
        smilesDetected=0
            
        
    if smilesDetected>1:
        imToShow = img    
    elif noSmiles>1:
        imToShow = desatBGR

        
        
    
            
    cv2.imshow('img',imToShow)
    out.write(imToShow)
    k = cv2.waitKey(30) & 0xff
    
    if k==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
        


# In[73]:


cv2.__version__


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:39:55 2019

@author: minja
"""

import torch
torch.cuda.is_available()
from torch.autograd import Variable
from utee import selector, misc, quant
import cv2
import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt

#%% Ovaj deo ce pokupiti potrebne podatke i model


model_raw, ds_fetcher, is_imagenet = selector.select('mnist') 

# podaci za validaciju. 
# nema podataka za train jer je model vec istreniran i ni u kom trenutku ti to neces morati da radis,
# interesuje te samo evaluation (test) faza
ds_val = ds_fetcher(batch_size=10,train=False,val=True)

for idx, (data,target) in enumerate(ds_val):
    data = Variable(torch.FloatTensor(data)).cuda() # ovo .cuda() salje podatke na GPU
    output = model_raw(data)
    
    #%%
    
print('Ovako izgleda model ',model_raw.model.__getattr__)


print('ds_val je tipa DataLoader duzine: ',ds_val.__len__())
# njemu je iznad receno da vraca batcheve podataka od po 10 slika i to smesta u promenljive data i target

# mozes da ga zoves iteratorom
#i = iter(ds_val)
#X,Y = next(i)
#print('10 slika: ', X)
#print('10 odgovarajucih oznaka: ', Y)
#print(X.shape)

# a to je zapravo i uradjeno gore kroz enumerate
print('velicina batcha', data.shape)

# ovde vidis da su slike ucitane u formatu CHW - iliti channels first, ali posto su grayscale dimenzija kanala je 1
# default redosled dimenzija se razlikuje od frameworka do frameworka, imati u vidu
# X i Y su klase tensor, koje mozes da prebacis u numpy arrays pomocu: X.numpy() i onda da radis sa njima sve sto numpy moze da ponudi
# p.s. neces moci sa ovim data zbog .cuda dodatka dok ne vratis na cpu (data.cpu().numpy())
# osim toga vidis da su ulazne dimenzije mreze 784 x 1, a tvoja slika je 28x28, sto znaci da mora da se flatten-uje prvo

#primer slike
plt.imshow(data[0,0,:,:].cpu().numpy(),cmap='gray')


#%%
# tezine modela su ovde:

w = list(model_raw.model.parameters())
print('ima ih ovoliko: ',len(w),' sto je manje od broja slojeva jer neki slojevi nemaju tezine, kao sto su aktivacije(relu) i dropout')

#primetices da se ni onda broj ne slaze
# to je zato sto jedan sloj koji ima tezine, kao fc1, zasebno ima tezine i biase,
# pa je drugi element liste zapravo deo prvog sloja (bias, po jedan za svaki izlazni neuron)


#%%
# za evaluaciju modela, moras da ga stavis u odgovarajuci mode:
model_raw.model.eval()
# ovo je bitno jer neke stvari, poput dropout koji nasumicno iskljucuje neurone, rade samo u trening fazi, a za test ne rade nista,
# medjutim ako ga ukljucis za test, on ce da unakazi rezultate

#with torch.no_grad():
#    Y_ = model_raw.model(data2) #<--ne radi zbog dimenzija. srecom neko je vec pisao fju (utee/misc.py/eval_model). 
    # Oni tu i prvo normalizuju sliku 

acc1, acc5 = misc.eval_model(model_raw, ds_val) # kad ja tamo, a ono ne radi. vraca prazne tenzore. izdebaguj!!
# kad proradi ( XD ) tu ces imati tacnost originalne mreze. 

#%% quantize weights

bits = 8 # ukupno bitova 

quantized_weights = []
for layer in w:
    sf = 4; 
    temp = quant.linear_quantize(layer,sf,bits)
    quantized_weights.append(temp)
    
#ucitaj nove tezine, tj napravi novu mrezu kvantizovanu
model_q = model_raw
#npr za 1 sloj:
model_q.model.fc1.weight.data = quantized_weights[0].clone().detach()
#...

# kad pokrenes eval na ovoj mrezi dobicecs tacnost kvantizovane mreze. Ova dva rezultata se zahteva da se razlikuju maksimalno za 3
# ako je veca razlika, probaj neku od ostalih fja za kvantizaciju
# ili, ja sam nasla da ume rezultat da bude bolji ako provozam nekoliko puta stochastic rounding (izguglaj)
# umesto floor koji ima u quantize fjama

#%% quantize activations

# za aktivacije mi nije ocigledno kako mogu da izvucem iz pretreniranog modela trenutno, 
# ima opcija da se ponovo sastavi mreza pa da fja za to vraca svaki sloj pojedinacno
# mozda ima laksi nacin, potrazicu
# a ne vidim ni da ovaj git koristi kvantizaciju aktivacija, ali mi mozda promice

# E sad, cemu ovo sluzi - aktivacije nemas gde da ucitas. One su prosto izlazi iz slojeva i zavise od podataka i to moras da radis on-the-go
# Medjutim ima zackoljica. Krenes od kvantizacije ulaznih podataka. 
# U slucaju da kvantizujes min-max metodom, aktivacije i tezine (i biasi) ce imati razlicit format
# (pogledaj: Q-format)
# proizvod ulaza i tezina daje izlaz koji je treceg formata
# i znanje o zeljenom Q formatu ti treba da bit-shiftujes akumulator da bi se sve slagalo







from math import sqrt
import pandas as pd
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import random
import matplotlib.pyplot as plt

# broj sesija u kojima je koriscena odgovarajuca kopmbinacija C
dataset={'Subject1':{'C5':4,'C49':1},
'Subject2':{'C1':5,'C6':7,'C8':1,'C22':1,'C34':4},
'Subject3':{'C1':3,'C7':10,'C9':2,'C22':1,'C37':4,'C44':3},
'Subject4':{'C2':10,'C8':3,'C10':1},
'Subject5':{'C3':2,'C8':7,'C11':2,'C38':5},
'Subject6':{'C38':5,'C34':4,'C5':2},
'Subject7':{'C41':8,'C42':2,'C50':1},
'Subject8':{'C1':12},
'Subject9':{'C47':5,'C48':3,'C50':2},
'Subject10':{'C2':1,'C10':7,'C12':3,'C13':1,'C23':6}}

# koja polja sadrzi koja kombinacija (0-ne sadrzi, 1-sadrzi)
datasetItemsDict={'C1':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C2':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C3':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C4':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':1,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C5':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':1,'Pad6':0,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C6':{'Pad1':0,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':0,'Pad9':1,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C7':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C8':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':1,'Pad5':0,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C9':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C10':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C11':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C12':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C13':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C14':{'Pad1':0,'Pad2':1,'Pad3':1,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C15':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':1,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C16':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':1,'Pad6':0,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C17':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C18':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':1,'Pad5':0,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C19':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C20':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C21':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C22':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C23':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C24':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':1,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C25':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':1,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C26':{'Pad1':0,'Pad2':0,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C27':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C28':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':0,'Pad5':1,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C29':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C30':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':1},
'C31':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C32':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C33':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':1,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C34':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':1,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C35':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C36':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C37':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':1,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C38':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':1,'Pad5':0,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C39':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':0,'Pad5':0,'Pad6':1,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C40':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':1},
'C41':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C42':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C43':{'Pad1':1,'Pad2':1,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C44':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':1,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C45':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':1,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C46':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':0,'Pad5':0,'Pad6':0,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':1,'Pad15':0,'Pad16':0},
'C47':{'Pad1':0,'Pad2':1,'Pad3':0,'Pad4':0,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':1,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
'C48':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':1,'Pad5':1,'Pad6':1,'Pad7':1,'Pad8':1,'Pad9':0,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':1,'Pad16':0},
'C49':{'Pad1':0,'Pad2':0,'Pad3':1,'Pad4':1,'Pad5':0,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':0,'Pad10':1,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':1,'Pad16':0},
'C50':{'Pad1':1,'Pad2':0,'Pad3':0,'Pad4':1,'Pad5':1,'Pad6':0,'Pad7':0,'Pad8':1,'Pad9':1,'Pad10':0,'Pad11':0,'Pad12':0,'Pad13':0,'Pad14':0,'Pad15':0,'Pad16':0},
}

datasetItems=np.transpose(pd.DataFrame(datasetItemsDict))


def clusterCombos(datasetItems,n_clusters):
	simMat=squareform(pdist(datasetItems,'jaccard'))
	model=AgglomerativeClustering(n_clusters=n_clusters,linkage='average',affinity='precomputed')
	model.fit(simMat)
	labels=datasetItems.index
	CompClust={} #Ovde su dakle imena kombinacija i njihovi odgovarajuci klasteri
	for i,comp in enumerate(labels):
		CompClust[comp]=model.labels_[i]
	return CompClust


def normalizeData(someData):
	data1=someData
	for subject in someData:
		tot=sum(someData[subject][comp] for comp in someData[subject])
		for comp in someData[subject]:
			data1[subject][comp]=someData[subject][comp]*100/tot;
	return data1


def euclidMyMan(sub1,sub2,data):
	bothRated={}
	for comp in data[sub1]:
		if comp in data[sub2]:
			bothRated[comp]=1;

	sizeOfOverlap=len(bothRated)
	if sizeOfOverlap==0:
		return 0

	s=sqrt(sum((data[sub1][comp]-data[sub2][comp])**2 for comp in bothRated))
	return (1/(1+s))

# def jaccardMyItem(padList1,padList2):
# 	union=len(set.union(padList1,padList2))
# 	intersect=len(set.intersection(padList1,padList2))
# 	return(intersect/union) #ne valja sa ovim 1 i 0 ovde

#print(jaccardMyItem(datasetItems(1),datasetItems(2))

# def similarDudes(sub,data,nUsers):
# 	assert nUsers<=len(data),\
# 		"Desired number of users exceeds the number of data entries"
# 	similarities=[(euclidMyMan(sub,subject,data),subject) for subject in data if subject!=sub]
# 	similarities.sort(reverse=True)
# 	return(similarities[0:nUsers])


def recommend(person,data):
	totals={}  #sums of score*similarity for all comps not already experienced by person
	similaritySums={} #sums of all similarities so you can divide
	# and compensate for different number of occurrence of certain comps
	for other in data:
		if other==person:
			continue
		eSim=euclidMyMan(person,other,data)
		if eSim==0:
			continue
		for comp in data[other]:
			if comp not in data[person]:
				totals.setdefault(comp,0)
				totals[comp]+=data[other][comp]*eSim
				similaritySums.setdefault(comp,0)
				similaritySums[comp]+=eSim
	rank=[(total/similaritySums[comp],comp) for comp,total in totals.items()]
	rank.sort(reverse=True)
	recommendList=[rec for score,rec in rank]
	return(recommendList)

def plotComp(datasetItemsDict,itemName):
	temp=datasetItemsDict[itemName]
	names=['Pad'+str(i) for i in range(1,17)]
	arr=[temp[name] for name in names]
	temp=np.reshape(arr,(4,4))
	plt.figure()
	plt.title(itemName)
	plt.imshow(temp)
	plt.colorbar()
	plt.show()
		
				

##################################### MAIN SHIZZ ############################
normData=normalizeData(dataset)

nClust=input('How many clusters do you want for your comps?: ')
CC=clusterCombos(datasetItems,int(nClust))


currentComp='C'+str(random.randint(1,50))
print('Your initial comp is: ',currentComp)
plotComp(datasetItemsDict,currentComp)
currentRating=input('Please rate this comp (0-Bad, 100-Perfect, 10-Needs tweaking): ')
normData['NewSubject']={currentComp:int(currentRating)}
print('New entry:', normData['NewSubject'])


# watch your currentComp and currentRating variables through the loop:
# Also watch what you add to normData
while int(currentRating)<100:
	recMeBaby=recommend('NewSubject',normData)
	if not recMeBaby:
		print('We found no similar persons at this moment.')
		theSameCluster=[comp for comp,val in CC.items() if val==CC[currentComp]]
		theOtherClusters=[comp for comp,val in CC.items() if val!=CC[currentComp]]
		if (currentRating=='10'):
			currentComp=random.choice(theSameCluster)

		else:
			currentComp=random.choice(theOtherClusters)
		
		print('#########################################')
		print('Here\'s an item_cluster-based suggestion: ',currentComp)

	else:
		print('I hereby recommend the following comps for you to try: ',recMeBaby)
		correspondingClusters=[CC[comp] for comp in recMeBaby]
		print('The corresponding clusters of those are: ',correspondingClusters)
		print('And the comp you just tried belongs to cluster: ',CC[currentComp])
		if (currentRating=='10'):
			if CC[currentComp] in correspondingClusters:
				bestFitIndex=correspondingClusters.index(CC[currentComp])
				currentComp=recMeBaby[bestFitIndex]
			else:
				currentComp=recMeBaby[0]
		else:
			for ind,item in enumerate(correspondingClusters):
				if item!=CC[currentComp]:
					currentComp=recMeBaby[ind]
					break
			else:
				currentComp=recMeBaby[0]
		print('#########################################')
		print('So I suggest you start with: ',currentComp)
	plotComp(datasetItemsDict,currentComp)
	currentRating=input('Please rate this comp (0-Bad, 100-Perfect, 10-Needs tweaking): ')
	normData['NewSubject'][currentComp]=int(currentRating)
	
	
if recMeBaby:
	print('We have a winner!',currentComp)	

	






import numpy as np
import serial
import timeit
from ctypes import windll
import time
import sys
import random

import sys
from PyQt4.QtGui import *
from PyQt4 import QtCore

########################################################################################
############################ FUNCTIONS DEFINITIONS  #############################
########################################################################################


stopIt=0

############ open serial port ##############

def openConnect(portName):
    
    # Define serial communication parameters
    ser = serial.Serial(port=portName, baudrate=921600, dsrdtr=False, xonxoff=False, rtscts=False)
     
    
    # Open communication
    if ser.isOpen() == False:        
        ser.open()
        print('Opening serial connection...')
    else:
        print('Port is already open.')
            
    return ser

############## close serial port #############

def closeConnect(ser):
    
    if ser.isOpen() == True:        
        ser.close()
        print('Closing serial connection... Goodbye!')
    else:
        print("The specified port is already closed.")



############# Parallel port marker ###############

# def ElectricalStimulation(trigger_up, port, p):
    
#         p.Out32(port, 128)  # put all high on port 2-9
#         #pin_out = int( '0000001', 2 ) 
# ​
#         time.sleep(trigger_up)
# ​
#         p.Out32(port, 0)  # put all low on port 2-9


###### Send messages passed as an array of ascii codes ######

def sendMessage(ser,mes):
    ser.write(mes)
    while ser.in_waiting<4:
        pass
    response=ser.read(ser.in_waiting)
    return response.decode("utf-8")

##### Create messages for turning ON and OFF the DC/DC converter ####

def getConverterOnMessage():
    
    return [ord(i) for i in ">ON<"]
    
def getConverterOffMessage():
    return [ord(i) for i in ">OFF<"]


######## Create message for setting the output voltage of the DC/DC converter #########

def getConverterVoltageMessage(volts):
        
    if volts>150:
        print("Attempt to set converter voltage too high")
        sys.exit(1)
    mes_ascii= [ord(i) for i in ">SV;x<"]
    mes_ascii[4]=volts
    
    return mes_ascii

##### Create message for STARTING and STOPPING stimulation (formerly trigger) #####

def getTriggerOnMessage():
    return [ord(i) for i in ">S<"]
def getTriggerOffMessage():
    return [ord(i) for i in ">R<"]


###### Create message for turning ON and OFF multiplexer mode #########

def getMuxOnMessage():
    return [ord(i) for i in ">MUX;ON<"]
def getMuxOffMessage():
    return [ord(i) for i in ">MUX;OFF<"]


####### Get frequency message (1-255Hz) ##########

def getFreqMessage(frequency):
    
    if frequency>255:
        print("Attempt to set frequency too high!")
        sys.exit(1)
     
    # Introducing the value 
    two_bytes = '0000000000000000'
    temp_list = list(two_bytes)
    temp_list[ 15 -len( bin(frequency)[2:] ) + 1:16 ] = bin(frequency)[2:]
    two_bytes = ''.join(temp_list)

    # Initialize command
    command = '>SF;xx<'

    # Convert to ascii
    frequency_ascii = [ord(i) for i in command]

    temp_list = list(frequency_ascii)

    # Dividing into bytes
    temp_list[4] = int(two_bytes[0 : 8],2)
    temp_list[5] = int(two_bytes[8 :],2)

    return temp_list

######### Create mesage for checking battery levels #########

def getBatteryLevelMessage():
    return [ord(i) for i in ">SOC<"]

######## Create message for Number of Pulses from a positive number or zero (continuous stimulation) ####

def getPulseNoMessage(noPulses):
    four_bytes = '00000000000000000000000000000000'
    temp_list = list(four_bytes)
    temp_list[ 31 -len( bin(noPulses)[2:] ) + 1:32 ] = bin(noPulses)[2:]
    four_bytes = ''.join(temp_list)
    
    # Initialize command
    command = '>SN;xxxx<'
    
    # Convert to ascii
    numplets_ascii = [ord(i) for i in command]         
    
    temp_list = list(numplets_ascii)
   
    # Dividing into bytes
    temp_list[4] = int( four_bytes[0 : 8], 2 )
    temp_list[5] = int( four_bytes[8 : 16], 2 )
    temp_list[6] = int( four_bytes[16 : 24], 2 )
    temp_list[7] = int( four_bytes[24 :], 2 )
    
    return temp_list

####### Create a message for Pulse width from a number between 50 and 1000 [us]  ########

def getPWMessage(pulseWidth):
    
    if pulseWidth>1000:
        print("Attempt to set pulsewidth too high!")
        sys.exit(1)
    
    # Introducing the value 
    two_bytes = '0000000000000000'
    temp_list = list(two_bytes)
    temp_list[ 15 -len( bin(pulseWidth)[2:] ) + 1:16 ] = bin(pulseWidth)[2:]
    two_bytes = ''.join(temp_list)
      
    # Initialize command
    command = '>PW;xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<'
    
    # Convert to ascii
    pw_ascii = [ord(i) for i in command]    
    
    
    # Dividing into bytes
    for pl in np.arange(16):
                          
            pw_ascii[2*(pl) +4] = int( two_bytes[0 : 8], 2 )
            pw_ascii[2*(pl) +5] = int( two_bytes[8 :], 2 )
            
    return pw_ascii

###### Create message for setting cathode-anode pairs ######

def getCathodeAnodeMessage(pletList):
    # input: a list of dictionaries containing cathodes (C) and anodes (A)
    # e.g. getCathodeAnodeMessge([{'C':[2,3,4],'A':[5,6,7,8]},{'C':[1],'A':[9,10]}])
    # This will set the first plet to C: 2,3,4 and A: 5,6,7,8; 
    # Second plet C: 1, A: 9,10
        
    command = '>CA;xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<'
    channels_ascii = [ord(i) for i in command]  
    
    pletIndex=0
    for plet in pletList:
        
        cathodeList=plet['C']
        anodeList=plet['A']
        
        i=0
        while  i<len(cathodeList) and cathodeList[i] not in anodeList:
            i+=1
        if i<len(cathodeList):
            print("!!!Attempt to set the same field as both cathode and anode!!!")
            sys.exit(1)
    
        # Define channels CATHODE
        channel_cathode = '0000000000000000'
        ch_list_cathode = list(channel_cathode)

        for ch in cathodeList:
            ch_list_cathode[16-ch] = '1'
        
        channel_cathode = ''.join(ch_list_cathode)
    
    
        # Define channels ANODE
        channel_anode = '0000000000000000'
        ch_list_anode = list(channel_anode)

        for ch in anodeList:
            ch_list_anode[16-ch] = '1'
        
        channel_anode = ''.join(ch_list_anode)          
    
       
        channels_ascii[4+pletIndex*4]=int(channel_cathode[0:8],2)
        channels_ascii[5+pletIndex*4]=int(channel_cathode[8:],2)
        channels_ascii[6+pletIndex*4]=int(channel_anode[0:8],2)
        channels_ascii[7+pletIndex*4]=int(channel_anode[8:],2)
        
        pletIndex+=1
    
    for i in range(pletIndex*4+4,68):
        channels_ascii[i]=int('00000000', 2)
    
    
    return (channels_ascii) 

######## Create message for setting current in MUX mode #####

def getCurrentMessage(currentList):
    
    # takes a list of currents to set, each element corresponding to one plet
    # e.g. getCurrentMesage([12,1,10])
    # enter the desired current intensity, the function will perform the 10x multiplication
    
    # Initialize command
    command = '>SC;xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<'
    current_ascii = [ord(i) for i in command]  
    
    currentList=10*np.array(currentList)
    
    # Assert all currents are smaller than 80mA
    if sum(currentList>800)>0:
        print("Cannot set current above 80mA.")
        sys.exit(1)
            
    # Create message
    pletIndex=0
    for current in currentList:
        
        # Introducing the value 
        two_bytes = '0000000000000000'
        temp_list = list(two_bytes)
        temp_list[ 15 -len( bin(current)[2:] ) + 1:16 ] = bin(current)[2:]
        two_bytes = ''.join(temp_list)
   
        current_ascii[2*(pletIndex) +4] = int( two_bytes[0 : 8], 2 )
        current_ascii[2*(pletIndex) +5] = int( two_bytes[8 :], 2 )
        
        pletIndex+=1
     
    # Fill the remaining places with zeros
    for i in range(pletIndex*2+4,36):
        current_ascii[i]=int('00000000', 2)

    return current_ascii
    
####### Get ASYNC message (out of MUX mode) ######

def getAsyncMessage():
    return [ord(i) for i in ">ASYNC<"]

##### Set active channels in ASYNC, out of MUX (sync is not reliable!) ####

def getSAMessage(cathodeList):
    
    # eg: mes=getSAMessage([1,2,3])
    # Enter Asynchronous mode
    r=sendMessage(ser,getAsyncMessage())
    print(r)
    
    # Initialize command
    command = '>SA;xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx<'
    channels_ascii = [ord(i) for i in command]    
    
    cathodeIndex=0
    for cathode in cathodeList:
        # Define channels
        channel_bin = '0000000000000000'
        ch_list = list(channel_bin)

        ch_list[16-cathode] = '1'
        
        channel_bin = ''.join(ch_list)  
                    
        channels_ascii[cathodeIndex*2+4] = int( channel_bin[0 : 8], 2 )
        channels_ascii[cathodeIndex*2+5] = int( channel_bin[8 :], 2 )
        cathodeIndex+=1
            
    for i in range(4+cathodeIndex*2,36):
        channels_ascii[i] = int( '00000000', 2 )
    
     
    return channels_ascii



###### H reflex #########

# Commence the H reflex current mapping procedure

def Href():
    
    
    initialCurrent=1                # mA
    endCurrent=50					#mA
    
    voltageOut=150           # volts
    PW=500                   # micro seconds
    num_pulses = 5
    
       
    # Set DC/DC output voltage 
    r=sendMessage(ser,getConverterVoltageMessage(voltageOut))
    print("Setting DC/DC output voltage to {}: {}".format(voltageOut,r))
    
    # Turn on the DC/DC converter
    r=sendMessage(ser,getConverterOnMessage())
    print("DC/DC on: ",r)
    
    # Set pulse width
    r=sendMessage(ser,getPWMessage(PW))
    print("PW {}: {}".format(PW,r))
    
    # Set number of pulses
    r=sendMessage(ser,getPulseNoMessage(num_pulses))
    print("Setting number of pulses to {}: {}".format(num_pulses,r))  
    
    # Enter multiplexing mode
    r=sendMessage(ser,getMuxOnMessage())
    print("MUX ON: ",r)

   
   
    for Hreflex_current in np.arange(initialCurrent, endCurrent, 1):

        ### Check if the stop button was clicked ###
        qApp.processEvents()
        if flagzie.stopped:
        	flagzie.stopped=False
        	btnStopHref.setChecked(False)
        	break

        # Set the current amplitude
        answer = sendMessage(ser,getCurrentMessage([Hreflex_current]))
        print('Setting current amplitude to {}: {}'.format(Hreflex_current,answer))
        
        # ElectricalStimulation(trigger_up_elec, port_address, p) # Send marker
    	
    	# Trigger the device
        answer = sendMessage(ser,getTriggerOnMessage())
        print('Trigger ON: ', answer)    
    
        time.sleep(np.random.uniform(8,10))
        
    # Turn off the converter
    r=sendMessage(ser,getConverterOffMessage())
    print("DC/DC off: ",r)
   
    return

###### BONESTIM AS AN ASSISTIVE TOOL #####

## Define the immutable parameters for stimulation: 

def BoneAssistParameters():
    
    voltageOut=150           # volts
    PW=300                   # micro seconds
    num_pulses = 0           # continuous
    frequency=30             # Hz
       
    # Set DC/DC output voltage 
    r=sendMessage(ser,getConverterVoltageMessage(voltageOut))
    print("Setting DC/DC output voltage to {}: {}".format(voltageOut,r))
    
    # Turn on the DC/DC converter
    r=sendMessage(ser,getConverterOnMessage())
    print("DC/DC on: ",r)
    
    # Set pulse width
    r=sendMessage(ser,getPWMessage(PW))
    print("PW {}: {}".format(PW,r))
    
    # Set number of pulses
    r=sendMessage(ser,getPulseNoMessage(num_pulses))
    print("Setting number of pulses to {}: {}".format(num_pulses,r))  
    
    # Enter multiplexing mode
    r=sendMessage(ser,getMuxOnMessage())
    print("MUX ON: ",r)

    #Set frequency
    r=sendMessage(ser,getFreqMessage(frequency))
    print("Frequency {}: {}".format(frequency,r))


        
###################################################################################################
########################################## MAIN ###################################################
###################################################################################################


# Open the serial port
portName='COM28'

ser=openConnect(portName)

### Signifies whether the STOP H REFLEX button was clicked or not ####
class flagClass:
	def __init__(self):
		self.stopped=False

flagzie=flagClass()


# # Define the parallel port
# print("Connecting to the parallel port...")
# port_address = 0xD050
# p = windll.inpoutx64

# # Trigger up
# trigger_up_elec = 0.01 



#####  Check the battery level ######
res=sendMessage(ser,getBatteryLevelMessage())
bat=[ord(r) for r in res]
print("BATTERY: ",bat[-2],"%")
if bat[-2]<10:
    print("Battery low. Please charge the stimulator.")



### start with these parameters as default ###
BoneAssistParameters()


################################################################################################
################################## PyQt4 GUI ###################################################
################################################################################################

###############################################################
#### Create app window ####

ap=QApplication(sys.argv)
w=QWidget()

w.resize(520,440)
w.setWindowTitle("BONESTIM TESTS")

################################################################
################## DEFINE CALLBACK FUNCTIONS ###################

def stopped():
	flagzie.stopped=True
	return

def onClickStim():
	current=[1]
	r=sendMessage(ser,getCurrentMessage(current))
	print("Current {}: {}".format(current,r))

	# run for defined time...

	r=sendMessage(ser,getTriggerOnMessage())
	print("Stimulation ON: ",r)
	print("Stimulating {}s...".format(int(lineEditStimDuration.text())))

	time.sleep(int(lineEditStimDuration.text()))
    
	r=sendMessage(ser,getTriggerOffMessage())
	print("Stimulation OFF: ",r)

	return


def changeCurrent():
 	cur=int(lineEditCurrent.text())
 	r=sendMessage(ser,getCurrentMessage([cur]))
 	print("Changing current to {}mA: {}".format(cur,r))
 	return 


def changeCA():
	
	CAtext=lineEditCA.text()

	cathodes=CAtext[CAtext.find('C')+4:CAtext.find('A')-3]
	cathodes=[int(i) for i in cathodes.split(',')]
	
	anodes=CAtext[CAtext.find('A')+4:-2]
	anodes= [int(i) for i in anodes.split(',')]

	CA=[{'C':cathodes,'A':anodes}]

	r=sendMessage(ser,getCathodeAnodeMessage(CA))
	print("Setting cathode-anode comp {}: {}".format(CA,r))


	return

def changePW():
	PW=int(lineEditPW.text())
	r=sendMessage(ser,getPWMessage(PW))
	print("Changing PW to {}us: {}".format(PW,r))
	return

def changeFrequency():
	f=int(lineEditFrequency.text())
	r=sendMessage(ser,getFreqMessage(f))
	print("Changing frequency to {}us: {}".format(f,r))
	return

def changeNoPulses():
	NP=int(lineEditNoPulses.text())
	r=sendMessage(ser,getPulseNoMessage(NP))
	print("Changing the number of pulses to {}: {}".format(NP,r))
	return

def setupBoneAssist():
	BoneAssistParameters()
	lineEditPW.setText("300")
	lineEditFrequency.setText("30")
	lineEditNoPulses.setText("0")
	return



#################################################################################
############################### Create GUI ######################################

### Create cathode anode input ###
lineEditCA=QLineEdit(w)
lineEditCA.editingFinished.connect(changeCA)
lineEditCA.setText("{'C':[1,2],'A':[3,4]}")

### Create current intensity input ###
lineEditCurrent=QLineEdit(w)
lineEditCurrent.setText("1")
lineEditCurrent.editingFinished.connect(changeCurrent)

### Create pulse width input ###
lineEditPW=QLineEdit(w)
lineEditPW.setText("300")
lineEditPW.editingFinished.connect(changePW)

### Create frequency input ###
lineEditFrequency=QLineEdit(w)
lineEditFrequency.setText("30")
lineEditFrequency.editingFinished.connect(changeFrequency)

### Create number of pulses input ###
lineEditNoPulses=QLineEdit(w)
lineEditNoPulses.setText("0")
lineEditNoPulses.editingFinished.connect(changeNoPulses)

### Create input for duration of stimulation ###
lineEditStimDuration=QLineEdit(w)
lineEditStimDuration.setText("5")

### Set parameters default for BONESTIM as assistive device ###
btnSetBoneAssistParameters=QPushButton("BoneSTIM Assist Parameters",w)
btnSetBoneAssistParameters.released.connect(setupBoneAssist)

### Stimulate continuously ###
btnStim=QPushButton("STIMULATE",w)
btnStim.setToolTip("This is for Bonestim as Assistive Device test")
btnStim.clicked.connect(onClickStim)

### Start H reflex loop - increment intensity by 1mA ###
btnHreflex=QPushButton('H reflex loop',w)
btnHreflex.released.connect(Href)


### Stop execution of the H reflex loop ###
btnStopHref=QPushButton("STOP H REF EXECUTION",w)
btnStopHref.clicked.connect(stopped)
btnStopHref.setCheckable(True)


### Exit the app window ##
btnExit=QPushButton("EXIT",w)
btnExit.setToolTip("This will close this window")
btnExit.move(350,350)
btnExit.released.connect(exit)


####################################### LAYOUT #####################################
formLayout = QFormLayout(w)
formLayout.addRow("Set CA pairs as {'C':list,'A':list}", lineEditCA)
formLayout.addRow("Set current (mA):", lineEditCurrent)
formLayout.addRow("Set pulse width (us):", lineEditPW)
formLayout.addRow("Set frequency (Hz):", lineEditFrequency)
formLayout.addRow("Set number of pulses:", lineEditNoPulses)
formLayout.addRow("Set stimulation duration (s):", lineEditStimDuration)
formLayout.addRow("RESET parameters for BoneSTIM as assistive device",btnSetBoneAssistParameters)
formLayout.addRow("Stimulate for BoneSTIM as assistive device",btnStim)
formLayout.addRow("Start the H reflex loop",btnHreflex)
formLayout.addRow("Break the loop (WAIT A BIT AFTER CLICKING)",btnStopHref)


########################## EXECUTE ##############################

w.show()
ap.exec_()


#################################################################################################

### CLose connection
closeConnect(ser)

####
	
k=input("press close to exit") 
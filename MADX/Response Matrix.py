#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import os, time
import psutil
import numpy as np


# In[82]:


activate=open('C://MADX/response_matrix//track_activate.txt','r') 
elements=open('C://MADX//response_matrix//elements.txt','r')
quadrupoles=open('C://MADX//response_matrix//quads.txt','r')

act=activate.readlines()
elem=elements.readlines()
quads=quadrupoles.readlines()


# In[83]:


## A program to get a Response Matrix. Here is launching the MADX.bat with variation of switched-on kickers to get BPM measurements.
## It includes 3 files: a main command file, a file with the structure of VEPP4M and a file with kickers.
## During program execution, the 1st file with an added kicker from the 3rd file is rewritten into an additional file.
## MADX.bat calls the additional file, which reads the 2nd file. The final structure is the structure from the 2nd file + a kicker from the 3rd file. 
## A Reponse Matrix is written into "response_matrix.txt"

## BPM measurements
terminate=open('C://MADX/response_matrix//track_terminate.txt','w') 
for line in act:
    terminate.writelines(line)
terminate.close()

os.startfile('C://MADX//response_matrix//madx.bat')
time.sleep(1)

BPM_readings=pd.read_csv('C://MADX//response_matrix//measure1.txt')
BPM_readings=BPM_readings.iloc[7:1339,0].astype(float)  # for all locations
#print(BPM_readings)
#df=df.iloc[7:1339,0].astype(float)-BPM_readings
#BPM_readings=BPM_readings.iloc[7:115,0].astype(float)   # for BPMs only
#BPM_readings=BPM_readings.iloc[7:61,0].astype(float)    # for BPMs only in 1 coordinate

#BPM_readings=open('C://MADX/response_matrix//measure1.txt','r') 
#measurements=[]
#k=0
#for line in BPM_readings:
    #if k>=8:
      #  measurements.append((line.split()))
   # k=k+1
#meas=[[float(y) for y in x] for x in measurements]

## Correctors
frames=[]
k=0

for element in elem:
    ## Rewritting a file
    terminate=open('C://MADX/response_matrix//track_terminate.txt','w') 
    for line in act:
        if line.startswith('use'):
            terminate.writelines(element.replace(";",",kick=0.00003;"))
        terminate.writelines(line)
        
    terminate.close()
    
    ## Calculating a Response Matrix
    os.startfile('C://MADX//response_matrix//madx.bat')
    time.sleep(1)
    #name="cmd.exe"
    #while name=="cmd.exe":
     #   for proc in psutil.process_iter():
     #       name=proc.name()
      #      if name=="cmd.exe":
      #          break
                
    df=pd.read_csv('C://MADX//response_matrix//measure1.txt')  
    df=df.iloc[7:1339,0].astype(float)-BPM_readings  # for all locations
    #df=df.iloc[7:115,0].astype(float)-BPM_readings    # for BPMs only
    #df=df.iloc[7:61,0].astype(float)-BPM_readings    # for BPMs only in 1 coordinate
    frames.append(df/0.00003)
    df1=df
print(df1)
"""

## Quadrupoles
for quad in quads:
    ## Rewritting a file
    terminate=open('C://MADX/response_matrix//track_terminate.txt','w') 
    for line in act:
        if line.startswith('use'):
            quad=quad.replace(";",'')
            quad=quad.split()
            index=quad.index('k1:=')+1
            gradient=float(quad[index])+0.001
            quad=(" ").join(quad[:index])
            terminate.writelines(quad.replace("k1:=","k1:="+str(gradient)+";"))
            print(quad.replace("k1:=","k1:="+str(gradient)+";"))
        terminate.writelines(line)
        
    terminate.close()
    
    ## Calculating a Response Matrix
    os.startfile('C://MADX//response_matrix//madx.bat')
    time.sleep(1)
    #name="cmd.exe"
    #while name=="cmd.exe":
     #   for proc in psutil.process_iter():
     #       name=proc.name()
      #      if name=="cmd.exe":
      #          break
                
    df=pd.read_csv('C://MADX//response_matrix//measure1.txt')  
    #df=df.iloc[7:1339,0].astype(float)-BPM_readings
    df=df.iloc[7:115,0].astype(float)-BPM_readings
    #df=df.iloc[7:61,0].astype(float)-BPM_readings
    frames.append(df/0.001)
    df1=df
"""
df=pd.concat(frames,axis=1)
df.to_csv('C://MADX//response_matrix//response_matrix.txt',index=False,header=False,sep="\t")


# In[84]:


#activate.close()
#elements.close()


# In[85]:


## To remove temporary files
#name="cmd.exe"
#while name=="cmd.exe":
#    for proc in psutil.process_iter():
 #       name=proc.name()
 #       if name=="cmd.exe":
 #           break
os.remove('C://MADX/response_matrix//track_terminate.txt')
#os.remove('C://MADX/response_matrix//measure1.txt')


# In[86]:


## Orbit correction
## Calculating an Inverse Response Matrix to get kickers' values and fields errors.
## 1.Write out Response Matrix with activated 1 field error and 1 kicker
## 2.Get BPM measurements with fields errors
## 3.

import pandas as pd
import numpy as np


kicks_2=np.zeros(20)
for i in range(0,1):
    ## BPMs measurements
    BPM_readings=pd.read_csv('C://MADX//response_matrix//measure1.txt')
    BPM_readings=BPM_readings.iloc[7:1339,0].astype(float)
    #BPM_readings=BPM_readings.iloc[7:115,0].astype(float)
    #kicks_2=kicks_1
    BPM=open('C://MADX/response_matrix//measure_bet_2test.txt','r') 
    measurements=[]
    k=0
    for line in BPM:
        if k>=8:
            measurements.append(float(line))
        k=k+1
    #print(measurements)
    #print(BPM_readings)
    BPM_readings=[measurements[i]-BPM_readings[7+i] for i in range(0,len(measurements))]  # for beta correction
    #print(BPM_readings)
    ## Response Matrix
    R_Matrix=pd.read_csv('C://MADX/response_matrix//response_matrix.txt',header=None,sep="\t")




    inv_matrix=np.linalg.pinv(R_Matrix)
    print("Inverse Response Matrix's shape:", inv_matrix.shape)
    print("Amount of BPMs:", len(BPM_readings))
    print("Amount of kickers:", inv_matrix.shape[0])
    kicks=inv_matrix.dot(BPM_readings)
    print("Kick values:", kicks)
    ##print(kicks[20])

    from scipy.optimize import minimize, least_squares,lsq_linear
    A = R_Matrix
    b=[x for x in BPM_readings]  # for beta correction
    #b=[-x for x in BPM_readings]
    print(df.shape,len(b))
    kicks_1=lsq_linear(A,b).x
    print(kicks_1)



    ## Rewritting a file
    terminate=open('C://MADX/response_matrix//track_terminate.txt','w') 
    for line in act:
        if line.startswith('use'):
            k=0
            
            for element in elem:
                kick=kicks_1[k]+kicks_2[k]
                terminate.writelines(element.replace(";",",kick= "+str(kick)+";"))
                k=k+1
            """
            for quad in quads:
                quad=quad.replace(";",'')
                quad=quad.split()
                index=quad.index('k1:=')+1
                gradient=float(quad[index])+kicks_1[k]+kicks_2[k]
                quad=(" ").join(quad[:index])
                terminate.writelines(quad.replace("k1:=","k1:= "+str(gradient)+";"+"\n"))
                k=k+1
            """
                

        terminate.writelines(line)

    terminate.close()
    os.startfile('C://MADX//response_matrix//madx.bat')
    time.sleep(1)
    kicks_2=kicks_2+kicks_1


# In[ ]:





# In[87]:


#os.startfile(r'C://Windows//system32//cmd.exe', '-start')
##import subprocess, keyboard,os,time
"""os.startfile(r'C://Windows//system32//cmd.exe')
time.sleep(1)
#keyboard.write('C://MADX//madx-win64-gnu.exe', exact=True, delay=0.01)
keyboard.send('enter')  # aliases: press_and_release
time.sleep(2)
keyboard.write("Wake up Neo... ", exact=True,delay=0.1)
keyboard.send('enter')
time.sleep(2)
keyboard.write("The Matrix has you... ", exact=True,delay=0.1)
keyboard.send('enter')
time.sleep(2)
keyboard.write("Follow Sana Fan... ", exact=True,delay=0.1)
time.sleep(2)
keyboard.send('exit')"""
#keyboard.write('call,file=C://MADX//response_matrix//track_activate;', exact=True, delay=0.01)
#keyboard.send('call,file=C://MADX//response_matrix//track_activate;')
#keyboard.press_and_release('enter')
#time.sleep(6)


# In[88]:


a="tl1: quadrupole, l:= 1.287, k1:= 0.23857;"
a=a.replace(";",'')
a=a.split()
index=a.index('k1:=')+1
gradient=float(a[index])+0.2
print(gradient)
a=(" ").join(a[:index])
print(a.replace("k1:=","k1:="+str(gradient)))


# In[89]:


#A = R_Matrix
#BPM=open('C://MADX/response_matrix//measure1.txt','r') 
#measurements=[]
#k=0
#for line in BPM:
 #   if k>=8:
  #      measurements.append(float(line))
   # k=k+1
#b=[-x for x in measurements]
#print(df.shape,len(b))
#kicks_1=[kicks_1[i]-lsq_linear(A,b,tol=1e-16).x[i] for i in range(0,len(kicks_1))]
#print(lsq_linear(A,b,tol=1e-16).x)
#print(kicks_1)


# In[90]:


#terminate=open('C://MADX/response_matrix//track_terminate.txt','w') 
#for line in act:
 #   if line.startswith('use'):
  #      k=0
   #     for element in elem:
    #        terminate.writelines(element.replace(";",",kick="+str(kicks_1[k])+";"))
     #       k=k+1
    #terminate.writelines(line)
        
#terminate.close()
#os.startfile('C://MADX//response_matrix//madx.bat')


# In[91]:


import numpy as np
a=[1,1,1]
b=np.ones(3)
c=np.zeros(3)
print(c+b)


# In[ ]:





# In[ ]:





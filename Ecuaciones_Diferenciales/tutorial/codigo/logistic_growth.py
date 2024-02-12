#!/usr/bin/env python
import numpy as np
import pylab as pl
"""
Simula el crecimiento de una población usando el algoritmo Gillespie
"""

b1 = 2.0
b2 = 1.0/1000.0
d1 = 1.0
d2 = 1.0/1000.0
ND=MaxTime=100.0;


INPUT = 2.0

def stoc_eqs(INP,ts):  
	Z=INP
	Rate1 = (b1-b2*Z)*Z
	Rate2 = (d1+d2*Z)*Z
	R1=pl.rand()
	R2=pl.rand()
	ts = -np.log(R2)/(Rate1+Rate2)
	if R1<(Rate1/(Rate1+Rate2)):
		Z += 1;  # birth
	else:
		Z -= 1;  # death
	return [Z,ts]

def Stoch_Iteration(INPUT):
	lop=0
	ts=0
	T=[0]
	RES=[0]
	while T[lop] < ND and INPUT > 0:
		[res,ts] = stoc_eqs(INPUT,ts)
		lop=lop+1
		T.append(T[lop-1]+ts)
		RES.append(INPUT)
		lop=lop+1
		T.append(T[lop-1])
		RES.append(res)
		INPUT=res
	return [RES, T]

[RES,t]=Stoch_Iteration(INPUT)

t=np.array(t)
RES=np.array(RES)
### plotting
np.savetxt('time.txt',t)
np.savetxt('data.txt',RES)
pl.plot(t/365., RES, 'r')
pl.show()

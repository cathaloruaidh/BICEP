import cProfile
import csv
import getopt
import gzip
import logging
import math
import multiprocessing
import os
import pprint
import re
import sys
import threading

import numpy as np
import pandas as pd
import scipy.special as sp


from cyvcf2 import VCF
from functools import partial
from multiprocessing import cpu_count, Pool, Manager
from scipy.integrate import quad, dblquad
from threading import Lock





## global variables

# binomial coefficients

binomCoeff = [[]]




# printing lock
s_print_lock = Lock()





# thread-friendly printing
def s_print(*a, **b):
	with s_print_lock:
		print(*a, **b)



# lock initialiser
def lock_init(l):
	global lock
	lock = l




# define Pedigree class to hold all pedigree info 
class Pedigree:
	def __init__(self, famID, indID, dadID, mamID, sexID, pheID):
		self.nPeople = len(np.unique(indID))

		# save a copy of the input
		self.famID = famID
		self.indID = indID
		self.dadID = dadID
		self.mamID = mamID
		self.sexID = sexID
		self.pheID = pheID


		# code phenotypes as 0,1 for convenience
		self.phenotypeActual = pheID.astype(int)-1


		# indices for the parents
		self.dadIndex = np.zeros(self.nPeople, dtype=int)
		self.mamIndex = np.zeros(self.nPeople, dtype=int)


		# founder and descendant info
		self.founderIndex = np.array([]).astype(int)
		self.nFounder = 0
		
		self.descendantTable = np.full((self.nPeople, self.nPeople), -1)
		self.completed = np.zeros(self.nPeople, dtype=int)
		self.hasParents = np.full(self.nPeople, True)
		self.isParent = np.full(self.nPeople, False)
		self.children = np.empty((self.nPeople,),object)

		#self.children = [ [] for _ in range(self.nPeople) ]

		for i in range(self.nPeople):
			# get indices of parents
			try:
				self.dadIndex[i] = np.where(self.indID == self.dadID[i])[0][0]
			except IndexError:
				self.dadIndex[i] = -1

			try:
				self.mamIndex[i] = np.where(self.indID == self.mamID[i])[0][0]
			except IndexError:
				self.mamIndex[i] = -1


			# create a boolean for parental info
			if self.dadIndex[i] == -1 and self.mamIndex[i] == -1:
				self.hasParents[i] = False


			# identify and count the founders
			if np.char.equal(self.dadID[i], "0") and np.char.equal(self.mamID[i], "0"):
				self.founderIndex = np.append(self.founderIndex, int(i))
				self.nFounder += 1

			self.nonFounderIndex = np.array([ int(x) for x in range(self.nPeople) if x not in self.founderIndex ])

			# get children if any
			self.children[i] = [ x for x in range(self.nPeople) if self.indID[i] == self.mamID[x] or self.indID[i] == self.dadID[x] ]

			self.isParent[i] = True if len(self.children[i]) > 0 else False
	

		# set founders in desentant table
		for founder in self.founderIndex:
			self.descendantTable[founder,:] = np.zeros(self.nPeople)
			self.completed[founder] = 1


		# set parent-offspring in descendant table
		for i in range(self.nPeople):
			self.descendantTable[i,i] = 1
			if self.isParent[i]:
				for child in self.children[i]:
					self.descendantTable[child, i] = 1
				

#		# populate descendant table
#		while np.count_nonzero(self.descendantTable == -1) > 0:
#			#print(self.descendantTable)
#			for i in range(self.nPeople):
#				if self.completed[i]:
#					for child in self.children[i]:
#						for j in range(self.nPeople):
#							if self.descendantTable[i,j] == 1:
#								self.descendantTable[child,j] = 1
#
#						# founders already done, so no dadIndex = mamIndex = -1
#						if self.completed[self.dadIndex[child]] and self.completed[self.mamIndex[child]]:
#							for j in range(self.nPeople):
#								if self.descendantTable[child,j] == -1:
#									self.descendantTable[child,j] = 0
#							self.completed[child] = 1


		# populate descendant table
		oldCount = 0
		newCount = np.count_nonzero(self.descendantTable == -1)
		counter = 0

		# this should iterate over the number of generations
		while oldCount != newCount:
			completedParents = [ i for i in self.nonFounderIndex if self.completed[self.dadIndex[i]] == 1 and self.completed[self.mamIndex[i]] == 1 and self.completed[i] == 0 ]
			for child in completedParents:
				for j in range(self.nPeople):
					if self.descendantTable[child,j] == -1:
						self.descendantTable[child,j] = max(self.descendantTable[self.dadIndex[child],j], self.descendantTable[self.mamIndex[child],j])
				self.completed[child] = 1

			oldCount = newCount
			newCount = np.count_nonzero(self.descendantTable == -1)
			counter = counter + 1

		self.descendantTable[self.descendantTable == -1] = 0




		# make relationship matrix from pedigree data
		# NOTE: this is designed for non-consanguinous pedigrees

		df = pd.DataFrame({"IID" : self.indID, "FID" : self.dadID, "MID" : self.mamID})
		kin = np.full((self.nPeople, self.nPeople), -1.0)

		for i in range(self.nPeople):
			kin[i,i] = 1

		kin_pd = pd.DataFrame(kin, columns=df["IID"].tolist(), index=df["IID"].tolist())


		# set relatedness of parent/offspring pairs
		parents = list(set(df["FID"].loc[df["FID"] != "0"].tolist() + df["MID"].loc[df["MID"] != "0"].tolist()))
		for parent in parents:
			children = df["IID"].loc[(df["FID"].isin([parent])) | (df["MID"].isin([parent]))].tolist()

			for child in children:
				kin_pd.loc[parent,child] = kin_pd.loc[child,parent] = 0.5

				# set relatedness of full or half siblings
				if len(children) > 1:
					for sib in [ _ for _ in children if _ != child ]:
						if kin_pd.loc[child,sib] == -1 or kin_pd.loc[sib,child] == -1:
							if (df["FID"][df["IID"] == child].values[0] == df["FID"][df["IID"] == sib].values[0]) and (df["MID"][df["IID"] == child].values[0] == df["MID"][df["IID"] == sib].values[0]):
								kin_pd.loc[child,sib] = kin_pd.loc[sib,child] = 0.5

							elif (df["FID"][df["IID"] == child].values[0] == df["FID"][df["IID"] == sib].values[0]) or (df["MID"][df["IID"] == child].values[0] == df["MID"][df["IID"] == sib].values[0]):
								kin_pd.loc[child,sib] = kin_pd.loc[sib,child] = 0.25


		# set relatedness of other relationships by comparing to an individual's parents
		# keep repeating this until no new relatedness coefficients are set
		while True:
			kin_pd_tmp = kin_pd
			for i in range(self.nPeople):
				father = df["FID"].loc[i]
				mother = df["MID"].loc[i]

				if father != "0" and mother != "0":
					fIDX = df.loc[df["IID"] == father].index.values[0]
					mIDX = df.loc[df["IID"] == mother].index.values[0]

					for j in [ _ for _ in range(self.nPeople) if _ != i ]:
						if kin_pd.iloc[j, fIDX] > 0.0 and kin_pd.iloc[j, i] == -1.0:
							kin_pd.iloc[i, j] = kin_pd.iloc[j, i] = kin_pd.iloc[fIDX, j] / 2.0

						if kin_pd.iloc[j, mIDX] > 0.0 and kin_pd.iloc[j, i] == -1.0:
							kin_pd.iloc[i, j] = kin_pd.iloc[j, i] = kin_pd.iloc[mIDX, j] / 2.0
			if (kin_pd_tmp == kin_pd).to_numpy().all():
				break

		# if no relatedness is known by now, set it to zero
		kin_pd[kin_pd == -1.0] = 0.0

		self.kinshipMatrix = kin_pd.to_numpy()




# return a string representation of the genotypes
# missing is '.', absent is '0' and carrier is '1'
def genotypeString(vector):
	return re.sub(']', '', re.sub('\[', '', re.sub('-1', '.', re.sub(' ', '', "".join(map(str, vector))))))




# find the most recent common ancestor(s) for a given
# genotype string
def getMRCA(genotype, pedInfo):

	# get founders all carriers are descended from
	carrierIndex = [ x for x in range(pedInfo.nPeople) if genotype[x] == 1 ]
	carrFounderIndex = []

	if len(carrierIndex) == 0:
		return "NA"

	for x in range(pedInfo.nPeople):
		
		add = True
		if genotype[x] == 0:
			add = False
			continue

		for carrier in carrierIndex:
			if pedInfo.descendantTable[carrier,x] == 0:
				add = False
				break
		if add:
			carrFounderIndex.append(x)

	if len(carrFounderIndex) == 0:
		return "NA"



	MRCA = []

	descSub = pedInfo.descendantTable[carrFounderIndex,:][:,carrFounderIndex]


	for i in range(len(carrFounderIndex)):
		descSub[i,i] = 0

	if np.max(descSub) == 0:
		MRCA.extend(pedInfo.indID[carrFounderIndex])

	else:
		for i in carrFounderIndex:
			add = True
			for j in carrFounderIndex:
				if j in pedInfo.children[i]:
					add = False

			if add:
				MRCA.append(pedInfo.indID[i])


	if len(MRCA) == 0:
		return "NA"
	
	else:
		return '|'.join(MRCA)







# return how many aff/unaff carriers/non-carriers for a given
# genotype string and the phenotypes
def phenoCarriers(genotype, pedInfo, samples):

	for i in range(pedInfo.nPeople):
		if pedInfo.indID[i] not in samples:
			genotype[i] = -1

	k1 = k2 = l1 = l2 = m = 0
	for i in range(pedInfo.nPeople):
		if pedInfo.phenotypeActual[i] == 1:
			if genotype[i] == 1:
				k1 += 1
			elif genotype[i] == 0:
				l1 += 1
		else:
			if genotype[i] == 1:
				k2 += 1
			elif genotype[i] == 0:
				l2 += 1
		
		if genotype[i] == -1 and pedInfo.indID[i] in samples:
			m += 1
	
	return k1, l1, k2, l2, m



# given the founder vector, calculate the number of genotype states
# that will be generated
def numGenotypeStates(founderVector, pedInfo, currParent):

	numPotential = 0
	for i in range(len(founderVector)):
		
		if founderVector[i] < 0:
			logging.error("Input vector had a missing genotype")
			return 0

		elif founderVector[i] > 1:
			numPotential += 1

	if numPotential == 0:
		return 1

	if founderVector[currParent] == 0:
		return 1

	if founderVector[currParent] == 1 and len(pedInfo.children[currParent]) == 0:
		return 1


	count = 1

	for child in pedInfo.children[currParent]:
		if len(pedInfo.children[child]) == 0:
			if founderVector[child] > 1:
				count *= 2

		else:
			count *= numGenotypeStates(founderVector, pedInfo, child)


	if founderVector[currParent] == 1:
		return count 

	else:
		return count + 1






# given a genotype vector, find potential generations following
# the rare variant assumption
#def findGenerations(inputVector, genotypeStates, pedInfo):
def findGenerations(inputVector, founderVector, pedInfo):


	# get list of potential probands for the input variant
	proIndex = [ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 1 and inputVector[x] == 1 ]

	if len(proIndex) == 0:
		logging.debug("No probands identified")
		return 


	# initialise the founder vectors
	#founderVector = {}


	# get founders all probands are descended from
	carrierIndex = [ x for x in range(pedInfo.nPeople) if inputVector[x] == 1 ]
	carrFounderIndex = []


	for x in range(pedInfo.nPeople):
		
		add = True
		if inputVector[x] == 0:
			add = False
			continue

		for carrier in carrierIndex:
			if pedInfo.descendantTable[carrier,x] == 0:
				add = False
		if add:
			carrFounderIndex.append(x)


	if len(carrFounderIndex) == 0:
		logging.debug("No founders identified")
		return 



	# total number of permissible genotype states
	totalGenoStates = 0


	# for each founder, get the permissible unobserved genotypes
	for founder in carrFounderIndex:

		vector = inputVector.copy()

		# founder is a carrier
		vector[founder] = 1


		# all other founders (not just proband common founders) are non-carriers
		# note: other founders must have empty genotypes
		othFounderIndex = [ x for x in pedInfo.founderIndex.astype(int) if x != founder and vector[x] < 0 ]
		for oth in othFounderIndex:
			vector[oth] = 0


		count = 0
		for i in pedInfo.founderIndex:
			if vector[i] > 0:
				count += 1

		if(count > 1):
			return 
			

		# if an individual is a descendant of the founder and an ancestor of a
		# proband, they must be a carrier. Return zero if an individual is
		# known not to be a carrier

		fail = False
		for carrier in carrierIndex:
			for i in range(pedInfo.nPeople):
				if vector[i] > 0:
					continue
				
				if pedInfo.descendantTable[i, founder] and pedInfo.descendantTable[carrier, i]:
					if vector[i] == 0:
						fail = True
					else:
						vector[i] = 1
		if fail:
			continue


		# zero out children of non-carriers
		while True:
			vecTmp = vector.copy()
			for i in range(len(vector)):
				if pedInfo.hasParents[i] and vector[pedInfo.dadIndex[i]] == 0 and vector[pedInfo.mamIndex[i]] == 0 and vector[i] < 0:
					vector[i] = 0
			if (vector == vecTmp).all():
				break

		# zero out anyone who isn't related to the founder
		for i in range(pedInfo.nPeople):
			if pedInfo.kinshipMatrix[i, founder] == 0 and vector[i] < 0:
				vector[i] = 0




		# if one parent is a carrier, set the generation of the children
		while np.count_nonzero(vector == -1) > 0: 
			for i in pedInfo.nonFounderIndex:
				if vector[i] == -1:
					if vector[pedInfo.dadIndex[i]] == 0 and vector[pedInfo.mamIndex[i]] > 0:
						vector[i] = vector[pedInfo.mamIndex[i]] + 1
					if vector[pedInfo.mamIndex[i]] == 0 and vector[pedInfo.dadIndex[i]] > 0:
						vector[i] = vector[pedInfo.dadIndex[i]] + 1


		# finally, save this vector as the founderVector and find all potential genotype
		# combinations from the permissible unobserved genotypes
		founderVector[pedInfo.indID[founder]] = vector.copy()

		
		

	#for vector in founderVector.values():
	#	setGenerations(vector, genotypeStates, pedInfo)
	
	
	return 





def I_del(k1, k2, l1, l2):

	k = k1+k2
	l = l1+l2
	n = k+l

	sum = 0.0
	for i in range(l2 + 1):
		sum += binomCoeff[l2][i]*pow( -1.0, l2-i)/float( (l-i+1) * (n-i+2) * binomCoeff[n-i+1][k2] )

	return 2*sum




def I_del_alt(k1, k2, l1, l2):

	k = k1+k2
	l = l1+l2
	n = k+l

	return 1.0 / float( binomCoeff[k][k1]*(k+1) * binomCoeff[l][l1]*(l+1))





def I_del_linear(k1, k2, l1, l2):
	k = k1+k2
	l = l1+l2
	n = k+l

	sum = 0.0
	
	for i in range(l2+1 + 1):
	
		tmp_q = 0.0
		for q in range(k1+l+2-i + 1):
			
			tmp_r = 0.0
			for r in range(k2 + 1):

				if q == k1+l+2-i and r == 0:
					tmp_r += pow(-1.0, k2)*math.log(2.0)
					
				else:
					tmp_r += binomCoeff[k2][r]*pow(-1.0, k2-r)*(pow(2.0, k1+l+2-i-q+r) - 1.0)/float(k1+l+2-i-q+r)

			tmp_q += binomCoeff[k1+l+2-i][q]*pow(2.0, q)*pow(-1.0, k1+l+2-i-q) * tmp_r
			print(tmp_q)
		sum += binomCoeff[l2+1][i]*pow(-1.0, l2+1-i)/float( l+2-i ) * tmp_q
			

	return sum * 4.0




def I_del_linear_numeric(k1, k2, l1, l2):
	I = dblquad(lambda p, b: 4*(b**k1)*((1-b)**k2)*(p**l1)*((1-p)**(l2+1))/(2-b), 0, 1, lambda b: 0, lambda b: b)

	return I[0]




def I_del_beta_numeric(k1, k2, l1, l2, x):
	I = dblquad(lambda p, b: x*x*(b**(k1+x-1))*((1-b)**k2)*(p**l1)*((1-p)**(l2+x-1))/(1 - (1-b)**(x)), 0, 1, lambda b: 0, lambda b: b)

	return I[0]




def I_del_old(k1, k2, l1, l2):

	k = k1+k2
	l = l1+l2
	n = k+l

	sum = 0.0

	for i in range(k2+1):
		tmp_k = binomCoeff[k2][i]*pow(-1.0, k2-i)/float(k-i+1)

		tmp_l = 0

		if l2 > 0:
			for j in range(l2):
				tmp_l += binomCoeff[l2-1][j]*pow(-1.0, l2-1-j)*( (1.0/float(l-j)) - (1.0/float(n-i-j+1)) )

		else:
			for j in range(k-i+1):
				tmp_l += 1.0/float(l1+j+1)


		sum += tmp_k*tmp_l

	return sum




def I_neu(k1, k2, l1, l2):

	n = k1+k2+l1+l2

	return 1.0 / float( binomCoeff[n][k1+l1] * (n+1) )





def I_neu_beta(k1, k2, l1, l2, xa, ya):

	n = k1+k2+l1+l2

	return 1.0 / float( binomCoeff[n + xa + ya - 2][k1 + l1 + xa - 1] * (n + xa + ya - 1) )



def I_neu_numeric(k1, k2, l1, l2):
	I = quad(lambda a: (a**(k1+l1))*((1-a)**(k2+l2)), 0, 1)
	return I[0]





# calculate likelihood ratio for a given genotype vector
#@profile
def calculateBF(pedInfo, allBF, priorParams, inputData):

	# inner functions
	
	# given a genotype vector, set the generations and resolve 
	# into one or two putative child vectors, then recurse
	#@profile
	def setGenerations(vector):
		# define nonlocal variables
		nonlocal genotypeStates
		nonlocal pedInfo


		# get minimum of input vector greater than 1
		minGen = max(vector)
		minIndex = np.where(vector == minGen)[0][0] 

		for i in range(len(vector)):
			if vector[i] > 1 and vector[i] < minGen:
				minGen = vector[i]
				minIndex = i


		# if all genotypes are set and the proband is a carrier, 
		# add the vector to the list and return
		if minGen == 1 :
			genotypeStates.append(vector.copy())
			return


		# if the vector is empty or the proband is not a carrier, return
		if minGen == 0 :
			return


		# set the minimum potential genotype to zero and recurse
		subVec1 = vector.copy()
		subVec1[minIndex] = 0
		setGenerations(subVec1)



		# set the minimum potential genotype to one (if possible by inheritance) and recurse
		if pedInfo.hasParents[minIndex] and ( vector[pedInfo.dadIndex[minIndex]] == 1 or vector[pedInfo.mamIndex[minIndex]] == 1):
			subVec2 = vector.copy()
			subVec2[minIndex] = 1
			setGenerations(subVec2)

		return


	inheritanceProbability = np.array(
		[ [ [ 1.0, 0.0, 0.0 ], [ 0.5, 0.5, 0.0 ], [ 0.0, 1.0, 0.0 ] ], 
		[ [ 0.5, 0.5, 0.0 ], [ 0.25, 0.5, 0.25 ], [ 0.0, 0.75, 0.25 ] ], 
		[ [ 0.0, 1.0, 0.0 ], [ 0.0, 0.75, 0.25 ], [ 0.0, 0.0, 0.1 ] ] ]
	)

	inheritanceProbabilityDominant = np.array(
		[ [ [ 1.0, 0.0 ], [ 0.5, 0.5 ] ], 
		[ [ 0.5, 0.5 ], [ 0.0, 1.0 ] ] ]
	)



	# get ID string
	inputGenotype, name = inputData

	# get prior parameters
	priorCaus, priorNeut = priorParams


	# if we've already calculated it, return the value
	if name in allBF:
		return float(allBF[name][0])


	BF = 0.0

	numerator = 0.0
	denominator = 0.0


	founderVector = {}
	findGenerations(inputGenotype, founderVector, pedInfo)

	
	totalGenoStates = 0
	for founder, vector in founderVector.items():
		foundIdx = np.where(pedInfo.indID == founder)[0][0]
		totalGenoStates += numGenotypeStates(vector, pedInfo, foundIdx)
		

	# check if the genotype states array is likely to be greater than half the total space in RAM
	mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

	sizeGenoStates = totalGenoStates * sys.getsizeof(inputGenotype)
	sizeGenoStatesMB = round(sizeGenoStates / (1024**2), 3)
	sizeGenoStatesGB = round(sizeGenoStates / (1024**3), 3)

	if sizeGenoStates > mem_bytes / 2:
		msg = "the genotype states array is " + str(sizeGenoStatesGB) + "GB, ignoring" 
		logging.warning(msg)
		with lock:
			allBF[name] = [ 0.0, 0.0, 0.0, 0 ]
		return 0.0


	# get genotype states from the founder vectors
	msg = "number of genotype states for " + genotypeString(inputGenotype) + " is " + str(totalGenoStates) + " (" + str(sizeGenoStatesMB) + "MB)"
	logging.debug(msg)


	genotypeStates = []
	for vector in founderVector.values():
		setGenerations(vector)

	# sanity check for number of genotypes
	if len(genotypeStates) == 0:
		logging.debug("No genotype states identifed. ")
		with lock:
			allBF[name] = [ 0.0, 0.0, 0.0, 0 ]
		
		return 0.0


	# get convert to np array
	genotypeStates = np.asarray(genotypeStates, dtype=np.uint8)

	msg = "estim. size - " + str( round(len(genotypeStates) * sys.getsizeof(genotypeStates[0]) / (1024**2), 3) ) + "MB"
	logging.debug(msg)
	msg = "actual size - " + str( round(genotypeStates.nbytes / (1024**2), 3)) + "MB"
	logging.debug(msg)



	# calculate genotype configuration probabilities, and
	# calculate the numerator and denominator of the Bayes Factor
	genotypeProbabilities = np.zeros(len(genotypeStates))


	for i in range(len(genotypeStates)):
		p = 1.0
		for j in range(pedInfo.nPeople):
			if pedInfo.hasParents[j]:
				if genotypeStates[i][pedInfo.dadIndex[j]] == 1 or genotypeStates[i][pedInfo.mamIndex[j]] == 1:
					p = p / 2.0
		genotypeProbabilities[i] = p if p != 1.0 else 0.0

		nList = range(pedInfo.nPeople)
		

		k1 = k2 = l1 = l2 = 0
		for x in range(pedInfo.nPeople):
			if pedInfo.phenotypeActual[x] == 1:
				if genotypeStates[i][x] == 1:
					k1 += 1
				else:
					l1 += 1
			else:
				if genotypeStates[i][x] == 1:
					k2 += 1
				else:
					l2 += 1

		n  = k1+k2+l1+l2 
		
		# Causal model, prior distribution for parameters
		if priorCaus == "uniform":
			#print(genotypeString(genotypeStates[i]), "I_unif = ", I_del(k1, k2, l1, l2), "\t - \tP(G_F) = ", genotypeProbabilities[i])
			numerator = numerator + I_del(k1, k2, l1, l2)*genotypeProbabilities[i]
			#numerator = numerator + I_del_alt(k1, k2, l1, l2)*genotypeProbabilities[i]

		elif priorCaus == "linear":
			#print(genotypeString(genotypeStates[i]), "I_bet = ", I_del_beta_numeric(k1, k2, l1, l2, 7), "\t - \tP(G_F) = ", genotypeProbabilities[i])
			#numerator = numerator + I_del_beta_numeric(k1, k2, l1, l2, 11)*genotypeProbabilities[i]

			#print(genotypeString(genotypeStates[i]), "I_lin = ", I_del_linear_numeric(k1, k2, l1, l2), "\t - \tP(G_F) = ", genotypeProbabilities[i])
			numerator = numerator + I_del_linear_numeric(k1, k2, l1, l2)*genotypeProbabilities[i]
		
		else:
			logging.error("Prior distribution for parameters under causal model not known. ")


		# Neutral model, prior distribution for parameters
		if priorNeut == "uniform":
			#print(genotypeString(genotypeStates[i]), "I_neu = ", I_neu(k1, k2, l1, l2), "\t - \tP(G_F) = ", genotypeProbabilities[i])
			denominator = denominator + I_neu(k1, k2, l1, l2)*genotypeProbabilities[i]
		
		elif "," in priorNeut:
			if len(priorNeut.split(",")) != 2:
				msg = "Incorrect number of parameters for Beta distribution: " + priorNeut
				logging.error(msg)

			a,b = [ int(x) for x in priorNeut.split(",") ]
			denominator = denominator + I_neu_beta(k1, k2, l1, l2, a, b)*genotypeProbabilities[i]
		
		else:
			logging.error("Prior distribution for parameters under neutral model not known. ")



	if denominator == 0.0 :
		BF = 0.0
	else:
		BF = numerator/denominator


	# aquire the lock and save the data correct to 10 decimal places
	with lock:
		myList = [ BF, numerator, denominator, len(genotypeStates) ]
		allBF[name] = [ '%.10f' % elem for elem in myList ]



	return float(BF)




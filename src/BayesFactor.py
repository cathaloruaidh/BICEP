import cProfile
import csv
import getopt
import logging
import math
import multiprocessing
import os
import pprint
import re
import sys
import threading

import numpy as np
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
		self.sexID = sexID
		self.pheID = pheID


		# code phenotypes as 0,1 for convenience
		self.phenotypeActual = pheID.astype(np.int)-1


		# indices for the parents
		self.dadIndex = np.zeros(self.nPeople, dtype=int)
		self.mamIndex = np.zeros(self.nPeople, dtype=int)


		# founder and descendant info
		self.founderIndex = np.array([]).astype(int)
		self.nFounder = 0
		
		self.descendantTable = np.full((self.nPeople, self.nPeople), -1)
		self.completed = np.zeros(self.nPeople, dtype=int)
		self.hasParents = np.full(self.nPeople, True)
		self.children = np.empty((self.nPeople,),object)

		#self.children = [ [] for _ in range(self.nPeople) ]

		for i in range(self.nPeople):
			# get indices of parents
			try:
				self.dadIndex[i] = np.where(indID == dadID[i])[0][0]
			except IndexError:
				self.dadIndex[i] = -1

			try:
				self.mamIndex[i] = np.where(indID == mamID[i])[0][0]
			except IndexError:
				self.mamIndex[i] = -1


			# create a boolean for parental info
			if self.dadIndex[i] == -1 and self.mamIndex[i] == -1:
				self.hasParents[i] = False


			# identify and count the founders
			if np.char.equal(dadID[i], "0") and np.char.equal(mamID[i], "0"):
				self.founderIndex = np.append(self.founderIndex, int(i))
				self.nFounder += 1

			self.nonFounderIndex = np.array([ int(x) for x in range(self.nPeople) if x not in self.founderIndex ])

			# get children if any
			self.children[i] = [ x for x in range(self.nPeople) if indID[i] == mamID[x] or indID[i] == dadID[x] ]
	

		# set founders in desentant table
		for founder in self.founderIndex:
			self.descendantTable[founder,:] = np.zeros(self.nPeople)
			self.completed[founder] = 1


		# populate descendant table
		while np.count_nonzero(self.descendantTable == -1) > 0:
			for i in range(self.nPeople):
				self.descendantTable[i,i] = 1
				if self.completed[i]:
					for child in self.children[i]:
						self.descendantTable[child, i] = 1
						for j in range(self.nPeople):
							if self.descendantTable[i,j] == 1:
								self.descendantTable[child,j] = 1

						# founders already done, so no dadIndex = mamIndex = -1
						if self.completed[self.dadIndex[child]] and self.completed[self.mamIndex[child]]:
							for j in range (self.nPeople):
								if self.descendantTable[child,j] == -1:
									self.descendantTable[child,j] = 0
							self.completed[child] = 1




# return a string representation of the genotypes
# missing is '.', absent is '0' and carrier is '1'
def genotypeString(vector):
	return re.sub(']', '', re.sub('\[', '', re.sub('-1', '.', re.sub(' ', '', "".join(map(str, vector))))))





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


		# if all genotypes are set and the proband is a carrier, add the vector to the list and return
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





	# get ID string
	inputGenotype, name = inputData

	# get prior parameters
	priorCaus, priorNeut = priorParams

	#logging.debug(name)
	#print(name)

	# if we've already calculated it, return the value
	if name in allBF:
		return allBF[name][0]


	BF = 0.0

	numerator = 0.0
	denominator = 0.0


	founderVector = {}
	#findGenerations(inputGenotype, genotypeStates, pedInfo)
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

	#local_vars = list(locals().items())
	#for var, obj in local_vars:
	#	print(var, sys.getsizeof(obj))
	#print("\n")

	for i in range(len(genotypeStates)):
		p = 1.0
		for j in range(pedInfo.nPeople):
			if pedInfo.hasParents[j]:
				if genotypeStates[i][pedInfo.dadIndex[j]] == 1 or genotypeStates[i][pedInfo.mamIndex[j]] == 1:
					p = p / 2.0
		genotypeProbabilities[i] = p if p != 1.0 else 0.0

		nList = range(pedInfo.nPeople)
		
		#k1 = len([ x for x in nList if pedInfo.phenotypeActual[x] == 1 and genotypeStates[i][x] == 1 ])
		#k2 = len([ x for x in nList if pedInfo.phenotypeActual[x] == 0 and genotypeStates[i][x] == 1 ])
		#l1 = len([ x for x in nList if pedInfo.phenotypeActual[x] == 1 and genotypeStates[i][x] == 0 ])
		#l2 = len([ x for x in nList if pedInfo.phenotypeActual[x] == 0 and genotypeStates[i][x] == 0 ])

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

	#print("\n")	
	#print("num = ", numerator, "\t-\tdenom = ", denominator)
	#print("\n\n\n")	


	if denominator == 0.0 :
		#BF = float("inf")
		BF = 0.0
	else:
		BF = numerator/denominator


	# aquire the lock and save the data correct to 10 decimal places
	with lock:
		myList = [ BF, numerator, denominator, len(genotypeStates) ]
		allBF[name] = [ '%.10f' % elem for elem in myList ]



	return BF





# main function
def BF_main(args):


	# command line arguments
	nCores = 1
	inputFamFile = None
	inputVcfFile = None
	outputPrefix = None
	outputLog = None
	minAffecteds = 0
	priorCaus = "uniform"
	priorNeut = "uniform"





	if args.cores is not None:
		if int(args.cores) <= cpu_count():
			nCores = int(args.cores)

	if args.fam is not None:
		inputFamFile = args.fam

	if args.minAff is not None:
		minAffecteds = int(args.minAff)

	if args.output is not None:
		outputPrefix = args.output

	if args.vcf is not None:
		inputVcfFile = args.vcf

	if args.priorCaus is not None:
		priorCaus = args.priorCaus

	if args.priorNeut is not None:
		priorNeut = args.priorNeut




	# up recursion limit
	sys.setrecursionlimit(10000)



	# read contents of file into np array
	logging.info("Reading input FAM file")

	try:
		f = open(inputFamFile, newline='')
	except FileNotFoundError:
		msg = "Could not file input FAM file: " + inputFamFile
		logging.error(msg)
		sys.exit("Exiting ... ")
	else:
		reader = csv.reader(f, delimiter='\t')
		pedigreeFile = np.array(list(reader))



	# pedigree file sanity checks
	if pedigreeFile.shape[1] < 6:
		logging.error("Input FAM has too few columns.")
		sys.exit("Exiting ... ")
		


	# define inputs
	famID = np.array(pedigreeFile[:,0])
	indID = np.array(pedigreeFile[:,1])
	dadID = np.array(pedigreeFile[:,2])
	mamID = np.array(pedigreeFile[:,3])
	sexID = np.array(pedigreeFile[:,4])
	pheID = np.array(pedigreeFile[:,5])


	# save pedigree info and initialise
	pedInfo = Pedigree(np.unique(famID), indID, dadID, mamID, sexID, pheID)

	


	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")


	# get VCF file and variant/sample information
	vcf = VCF(inputVcfFile, gts012=True)
	nVariants = sum(1 for line in open(inputVcfFile) if not bool(re.match("^#", line)))
	vcfSampleIndex = []

	for i in range(len(vcf.samples)):
		try:
			ind = np.where(pedInfo.indID == vcf.samples[i])[0][0]
		except:
			msg = "Sample " + vcf.samples[i] + " is in VCF but not FAM."
			logging.warning(msg)
			sys.exit("Exiting ... ")
		else:
			vcfSampleIndex.append(ind)


	logging.info("Store as np array")

	# set all genotypes to missing as input
	genotypes = np.full((pedInfo.nPeople, nVariants), -1)


	# create list to hold unique variant ID
	varID = []


	# loop over all samples in VCF and get genotype
	j = 0
	for variant in vcf:
		# get the ID of the variant
		varID.append(variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0])


		# fill the known genotypes
		for i in range(len(vcfSampleIndex)):

			# get genotype type: {0,1,2,3}
			gt = int(variant.gt_types[i])

			# dominant inheritance: HET and HOM_ALT are the same. 
			# missing genotypes are set to -1
			if gt == 2:
				gt = 1

			if gt == 3:
				gt = -1

			# set known genotype
			genotypes[vcfSampleIndex[i]][j] = gt
		j += 1

	varString = np.apply_along_axis(genotypeString, 0, genotypes)

	# transpose array for parallelisation
	genotypes = np.transpose(genotypes.astype(np.int))


	# combine variant name with genotypes
	data = [ (genotypes[i],varString[i]) for i in range(len(genotypes)) ]


	################################################################################
	# set global variables
	################################################################################

	global binomCoeff
	binomCoeff = [ [0]*(pedInfo.nPeople + 20) for _ in range(pedInfo.nPeople + 20) ]

	for i in range(pedInfo.nPeople + 20):
		for j in range(i+1):
			binomCoeff[i][j] = float(sp.binom(i,j))




	################################################################################
	# calculate Bayes factors
	################################################################################



	# create dictionary to store all BF
	manager = Manager()
	allBF = manager.dict()




	if minAffecteds > 0:
		msg = "Removing variants with minAff < " + str(minAffecteds)
		logging.info(msg)

		for i in range(len(genotypes)):

			count = 0
			affs = [ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 1 ]

			for aff in affs:
				if genotypes[i][aff] == 1:
					count += 1

			if count < minAffecteds:
				allBF[varString[i]] = [ 0.0, 0.0, 0.0, 0 ]

				msg = "Removed variant: " + varString[i]
				logging.debug(msg)



	logging.info("Calculating Bayes Factors")

	# calculate the Bayes Factor for all variants
	if nCores > 1:
		# partial function for parallelisation - all constant except the input genotypes
		func = partial(calculateBF, pedInfo, allBF, [priorCaus, priorNeut])

		# create multiprocessing pool with lock
		l = multiprocessing.Lock()
		pool = Pool(nCores, initializer=lock_init, initargs=(l,))
		BFs = pool.map(func, data)
		pool.close()

	else:
		l = multiprocessing.Lock()
		lock_init(l)

		BFs = []
		for i in range(len(genotypes)):
			BFs.append(calculateBF(pedInfo, allBF, [priorCaus, priorNeut], data[i]))
	

	#results = [ '%.6f' % float(elem) for elem in BFs ]

	#print(float(results[1:10]))


	################################################################################
	# Output
	################################################################################

	logging.info("Output")

	
	if outputPrefix is None:
		for i in range(len(varID)):
			print(varID[i], "\t", BFs[i], "\t", varString[i], "\t")
	else:
		with open(args.outputDir + outputPrefix + ".BF.txt", 'w') as f:
			print("ID\tBF\tlogBF\tSTRING", file=f)
			for i in range(len(varID)):
				print(varID[i], "\t", BFs[i], "\t", np.log10(float(BFs[i])), "\t", varString[i], file=f)
		






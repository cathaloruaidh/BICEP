#!/usr/bin/python

# Conversion of CoSeg R package to Python



import cProfile, csv, getopt, logging, math, multiprocessing, os, pprint, re, sys, threading
import numpy as np
import scipy.special as sp

from cyvcf2 import VCF
from functools import partial
from multiprocessing import Pool, Manager
from threading import Lock


s_print_lock = Lock()


# thread-friendly printing
def s_print(*a, **b):
	with s_print_lock:
		print(*a, **b)


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

	

		# set founders in desentant table
		for founder in self.founderIndex:
			self.descendantTable[founder,:] = np.zeros(self.nPeople)
			self.completed[founder] = 1


		# populate descendant table
		while np.count_nonzero(self.descendantTable == -1) > 0:
			for i in range(self.nPeople):
				self.descendantTable[i,i] = 1
				if self.completed[i]:
					childrenIndex = [ x for x in range(self.nPeople) if indID[i] == mamID[x] or indID[i] == dadID[x] ]
					for child in childrenIndex:
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
def	genotypeString(vector):
	return np.array2string(vector, separator="").replace(" ", "").replace("-1", ".").replace("[", "").replace("]", "")




# given a genotype vector, resolve into one or two putative child vectors
# and recurse
def findGenotypes(vector, genotypeStates, pedInfo):

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
	findGenotypes(subVec1, genotypeStates, pedInfo)



	# set the minimum potential genotype to one (if possible by inheritance) and recurse
	if pedInfo.dadIndex[minIndex] >= 0 and pedInfo.mamIndex[minIndex] >= 0 and ( vector[pedInfo.dadIndex[minIndex]] == 1 or vector[pedInfo.mamIndex[minIndex]] == 1):
		subVec2 = vector.copy()
		subVec2[minIndex] = 1
		findGenotypes(subVec2, genotypeStates, pedInfo)

	return






# check if two vectors are the same, allowing for missing values
def matchVectors(vectorQuery, vectorTarget):

	# sanity check: the vectors have to have the same length
	if len(vectorQuery) != len(vectorTarget):
		logging.debug("In matchVectors, vectors have different lengths")
		return False


	# loop through and check identity, ignoring sites that are missing (negative values)
	for i in range(len(vectorQuery)):
		if vectorQuery[i] < 0:
			continue
		if vectorQuery[i] != vectorTarget[i]:
			return False

	return True




def I_del(k1, k2, l1, l2):

	k = k1+k2
	l = l1+l2
	n = k+l

	sum = 0.0
	for i in range(k2+1):
		tmp_k = float(sp.binom(k2, i))*pow(-1.0, k2-i)/float(k-i+1)

		tmp_l = 0

		if l2 > 0:
			for j in range(l2):
				tmp_l += float(sp.binom(l2-1, j))*pow(-1.0, l2-j-1)*( (1.0/float(l-j)) - (1.0/float(n-i-j+1)) )

		else:
			for j in range(k-i+1):
				tmp_l += 1.0/float(l1+j+1)


		sum += tmp_k*tmp_l

	return sum




def I_neu(k1, k2, l1, l2):

	n = k1+k2+l1+l2

	return 1.0 / float( sp.binom(n+1, k1+l1) * (k1+l1+1) )









# calculate likelihood ratio for a given genotype vector
def calculateBF(pedInfo, allBF, inputGenotype):

	# get ID string
	name=genotypeString(inputGenotype)

	# if we've already calculated it, return the value
	if name in allBF:
		return allBF[name][0]


	BF = 0.0

	numerator = 0.0
	denominator = 0.0


	# get list of potential probands for the input variant
	proIndex = [ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 1 and inputGenotype[x] == 1 ]

	if len(proIndex) == 0:
		allBF[name] = [ 0.0, 0.0, 0.0 ]
		return 0.0


	# initialise the founder vectors
	founderVector = {}
	genotypeStates = []



	# get founders all probands are descended from
	proFounderIndex = []

	for x in range(pedInfo.nPeople):
		
		add = True
		if inputGenotype[x] == 0:
			add = False
			continue

		for pro in proIndex:
			if pedInfo.descendantTable[pro,x] == 0:
				add = False
		if add:
			proFounderIndex.append(x)



	# for each founder, get the permissible unobserved genotypes
	for founder in proFounderIndex:
		vector = inputGenotype.copy()

		# founder is a carrier
		vector[founder] = 1


		# all other founders (not just proband common founders) are non-carriers
		# note: other founders must have empty genotypes
		othFounderIndex = [ x for x in pedInfo.founderIndex.astype(int) if x != founder and vector[x] < 0 ]
		for oth in othFounderIndex:
			vector[oth] = 0


		count = 0
		for founder in pedInfo.founderIndex:
			if vector[founder] > 0:
				count += 1

		if(count > 1):
			allBF[name] = [ 0.0, 0.0, 0.0 ]
			return 0.0
			

		# if an individual is a descendant of the founder and an ancestor of a
		# proband, make them a carrier. Ignore if genotype is non-missing
		for pro in proIndex:
			for i in range(pedInfo.nPeople):
				if vector[i] >= 0:
					continue
				elif pedInfo.descendantTable[i, founder] and pedInfo.descendantTable[pro, i]:
					vector[i] = 1


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

		findGenotypes(vector, genotypeStates, pedInfo)

	


	# sanity check for number of genotypes
	if len(genotypeStates) == 0:
		logging.warning("Error: no genotypes found! ")		

		allBF[name] = [ 0.0, 0.0, 0.0 ]
		return 0.0


	# get unique genotypes
	genotypeStates = np.unique(np.asarray(genotypeStates), axis=0)



	# calculate genotype configuration probabilities
	genotypeProbabilities = np.zeros(len(genotypeStates))

	for i in range(len(genotypeStates)):
		p = 1.0
		for j in range(pedInfo.nPeople):
			if pedInfo.hasParents[j]:
				if genotypeStates[i][pedInfo.dadIndex[j]] == 1 or genotypeStates[i][pedInfo.mamIndex[j]] == 1:
					p = p / 2.0
		genotypeProbabilities[i] = p if p != 1.0 else 0.0





	# get indices of matching
	genotypeMatchIndex = [ x for x in range(len(genotypeStates)) if matchVectors(inputGenotype, genotypeStates[x]) ]


	for index in genotypeMatchIndex:

		k1 = len([ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 1 and genotypeStates[index][x] == 1 ])
		k2 = len([ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 0 and genotypeStates[index][x] == 1 ])
		l1 = len([ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 1 and genotypeStates[index][x] == 0 ])
		l2 = len([ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 0 and genotypeStates[index][x] == 0 ])
		n  = k1+k2+l1+l2 
		numerator = numerator + I_del(k1, k2, l1, l2)*genotypeProbabilities[index]
		denominator = denominator + I_neu(k1, k2, l1, l2)*genotypeProbabilities[index]/float(k1+l1)
	

	if denominator == 0 :
		BF = float("inf")
	else:
		BF = numerator/denominator

	myList = [ BF, numerator, denominator ]	
	allBF[name] = [ '%.10f' % elem for elem in myList ]

	#print(multiprocessing.current_process(), " - ", name)


	return BF






# main function
def main(argv):


	# command line arguments
	inputFamFile = None
	inputVcfFile = None
	p1 = 0.01
	p2 = 0.8
	nCores = 1

	try:
		opts, args = getopt.getopt(argv, "c:f:l:p:v:", ["cores=", "fam=", "log=", "proband=", "vcf=", "prob1=", "prob2="])
	except getopt.GetoptError:
		print("Getopt Error")
		logging.error("getopt error")
		sys.exit("Exiting ... ")

	for opt, arg in opts:
		if opt in ("-c", "--cores"):
			if int(arg) <= multiprocessing.cpu_count():
				nCores = int(arg)

		if opt in ("-f", "--fam"):
			inputFamFile = arg
	
		if opt in ("-l", "--log"):
			logLevel = arg.upper()
			numeric_level = getattr(logging, arg.upper(), None)
			if not isinstance(numeric_level, int):
				raise ValueError('Invalid log level: %s' % arg)

		if opt in ("-v", "--vcf"):
			inputVcfFile = arg
	
	FORMAT = '# %(asctime)s [%(levelname)s] - %(message)s'
	
	try:
		logLevel
	except:
		logging.basicConfig(format=FORMAT)
	else:
		numeric_level = getattr(logging, logLevel, None)
		if not isinstance(numeric_level, int):
			raise ValueError('Invalid log level: %s' % logLevel)
		logging.basicConfig(format=FORMAT, level=logLevel)



	# up recursion limit
	sys.setrecursionlimit(10000)



	# read contents of file into np array
	logging.info("Reading input FAM file")

	try:
		f = open(inputFamFile, newline='')
	except FileNotFoundError:
		mesg = "Could not file input FAM file: " + inputFamFile
		logging.error(mesg)
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




	# define constants






	################################################################################
	# set pedigree variables
	################################################################################

	logging.info("Setting pedigre variables")




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
			mesg = "Sample " + vcf.samples[i] + " is in VCF but not FAM."
			logging.warning(mesg)
			sys.exit("Exiting ... ")
		else:
			vcfSampleIndex.append(ind)



	# set all genotypes to missing as input

	genotypes = np.full((pedInfo.nPeople, nVariants), -1)


	# create list to hold inuqie variant ID
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


	# create dictionary to store all BF
	manager = Manager()
	allBF = manager.dict()


	logging.info("Calculating Bayes Factors")


	# calculate the Bayes Factor for all variants
	if nCores > 1:
		# partial function for parallelisation - all constant except the input genotypes
		func = partial(calculateBF, pedInfo, allBF)

		# create multiprocessing pool 
		pool = Pool(nCores)
		BFs = pool.map(func, genotypes)
		pool.close()

	else:
		BFs = []
		for i in range(len(genotypes)):
			BFs.append(calculateBF(pedInfo, allBF, genotypes[i]))


	results = [ '%.6f' % float(elem) for elem in BFs ]


	################################################################################
	# Output
	################################################################################

	logging.info("Output")

	
	for i in range(len(varID)):
		print(varID[i], "\t", results[i], "\t", varString[i], "\t", varString[i].count("."))
		#print(results[i], "\t", varString[i], "\t", genotypes[i])

	#pprint.pprint(dict(allBF))







if __name__ == "__main__":
	main(sys.argv[1:])

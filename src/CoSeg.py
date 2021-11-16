#!/usr/bin/python

# Conversion of CoSeg R package to Python



import sys, csv, getopt, logging, cProfile, pprint, re, multiprocessing, threading
from multiprocessing import Pool, Manager
from threading import Lock
from functools import partial
from cyvcf2 import VCF
import numpy as np


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
	return np.array2string(vector, separator="").replace(" ", "").replace("-9", ".").replace("[", "").replace("]", "")




# for a given genotype, find the individuals who are likely to be the 
# most recent common ancestor via whom the variant was transmitted
def findGenotypeFounders(vector, pedInfo):
	genotypeFounders = []


	# find all individuals who are ancestors of all individuals within the family who have a genotype
	for i in range(pedInfo.nPeople):
		founder = True
		for j in range(pedInfo.nPeople):
			if i == j:
				continue
			elif vector[j] >= 0 and pedInfo.hasParents[j] and pedInfo.descendantTable[j, i] != 1:
				founder = False

		if founder:
			genotypeFounders.append(i)


	# remove any indeividuals who are ancestors of other founders
	forDel = []

	for i in genotypeFounders:	
		desc = False
		for j in genotypeFounders:
			if i != j and pedInfo.descendantTable[j,i] == 1:
				desc = True
		if desc:
			forDel.append(i)

	for f in forDel:
		genotypeFounders.remove(f)




	return genotypeFounders




# given a genotype vector, resolve into one or two putative child vectors
# and recurse
def findGenotypes(vector, genotypeStates, pedInfo, proIndex):

	# zero out individuals whose parents are non-carriers
	for i in range(len(vector)):
		if (pedInfo.dadIndex[i] >= 0 and pedInfo.mamIndex[i] >= 0):
			if vector[pedInfo.dadIndex[i]] == 0 and vector[pedInfo.mamIndex[i]] == 0:
				vector[i] = 0


	# get minimum of input vector greater than 1
	minGen = max(vector)
	minIndex = np.where(vector == minGen)[0][0] 

	for i in range(len(vector)):
		if vector[i] > 1 and vector[i] < minGen:
			minGen = vector[i]
			minIndex = i


	# if all genotypes are set and the proband is a carrier, add the vector to the list and return
	if ( minGen == 1 and vector[proIndex] == 1):
		genotypeStates.append(vector.copy())
		return


	# if the vector is empty or the proband is not a carrier, return
	if ( minGen == 0 or vector[proIndex] == 0):
		return



	# set the minimum potential genotype to zero and recurse
	subVec1 = vector.copy()
	subVec1[minIndex] = 0
	findGenotypes(subVec1, genotypeStates, pedInfo, proIndex)



	# set the minimum potential genotype to one (if possible by inheritance) and recurse
	if pedInfo.dadIndex[minIndex] >= 0 and pedInfo.mamIndex[minIndex] >= 0 and ( vector[pedInfo.dadIndex[minIndex]] == 1 or vector[pedInfo.mamIndex[minIndex]] == 1):
		subVec2 = vector.copy()
		subVec2[minIndex] = 1
		findGenotypes(subVec2, genotypeStates, pedInfo, proIndex)

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




# calculate likelihood ratio for a given genotype vector
def calculateLR(pedInfo, proIndex, phenotypeProbability, allLR, inputGenotype):

	epsilon = 0.001
	
	# get ID string
	name=genotypeString(inputGenotype)


	# if we've already calculated it, return the value
	if name in allLR:
		return allLR[name][0]




	# initialise the founder vectors
	founderVector = {}
	genotypeStates = []


	# get founders proband is descended from
	proFounderIndex = [ x for x in pedInfo.founderIndex if pedInfo.descendantTable[proIndex,x] == 1 ]


	for founder in proFounderIndex:
		vector = np.full(pedInfo.nPeople, -1)

		# founder is a carrier
		vector[founder] = 1


		# proband is a carrier
		vector[proIndex] = 1


		# other founders are non-carriers
		othFounderIndex = [ x for x in pedInfo.founderIndex.astype(int) if x != founder ]
		for oth in othFounderIndex:
			vector[oth] = 0


		# if an individual is a descendant of the founder and an ancestor of a
		# proband, make them a carrier. Ignore if genotype is non-missing
		for i in range(pedInfo.nPeople):
			if vector[i] >= 0:
				continue
			elif pedInfo.descendantTable[i, founder] and pedInfo.descendantTable[proIndex, i]:
				vector[i] = 1


		# zero out children of non-carriers
		while True:
			sumVec = np.sum(vector)
			for i in range(len(vector)):
				if  pedInfo.hasParents[i] and vector[pedInfo.dadIndex[i]] == 0 and vector[pedInfo.mamIndex[i]] == 0:
					vector[i] = 0
			if sumVec == np.sum(vector):
				break

		# children of non-carrier founders are non-carriers
		# if one parent is a carrier, set the generation of the children
		while np.count_nonzero(vector == -1) > 0: 
			for i in pedInfo.nonFounderIndex:
				if vector[i] == -1:
					if vector[pedInfo.dadIndex[i]] == 1 and vector[pedInfo.mamIndex[i]] == 0:
						vector[i] = 2
					if vector[pedInfo.mamIndex[i]] == 1 and vector[pedInfo.dadIndex[i]] == 0:
						vector[i] = 2
					if (vector[pedInfo.dadIndex[i]] == 1 or vector[pedInfo.dadIndex[i]] == 0) and vector[pedInfo.mamIndex[i]] > 1:
						vector[i] = vector[pedInfo.mamIndex[i]] + 1
					if (vector[pedInfo.mamIndex[i]] == 1 or vector[pedInfo.mamIndex[i]] == 0) and vector[pedInfo.dadIndex[i]] > 1:
						vector[i] = vector[pedInfo.dadIndex[i]] + 1
		founderVector[pedInfo.indID[founder]] = vector.copy()
		findGenotypes(vector, genotypeStates, pedInfo, proIndex)

	
	if len(genotypeStates) == 0:
		logging.warning("Error: no genotypes found! ")
		sys.exit("Exiting ... ")


	genotypeStates = np.unique(np.asarray(genotypeStates), axis=0)


	#print(name, ", ", len(genotypeStates))
	#print(inputGenotype)
	#pprint.pprint(genotypeStates)





	# calculate genotype configuration probabilities
	genotypeProbabilities = np.zeros(len(genotypeStates))

	for i in range(len(genotypeStates)):
		p = 1.0
		for j in range(pedInfo.nPeople):
			if pedInfo.dadIndex[j] >= 0 and pedInfo.mamIndex[j] >= 0:
				if genotypeStates[i][pedInfo.dadIndex[j]] == 1 or genotypeStates[i][pedInfo.mamIndex[j]] == 1:
					p = p / 2.0
		genotypeProbabilities[i] = p if p != 1.0 else 0.0





	# get indices of matching
	genotypeMatchIndex = [ x for x in range(len(genotypeStates)) if matchVectors(inputGenotype, genotypeStates[x]) ]
#	queryVector =  np.zeros(pedInfo.nPeople)
#	genotypeIndex = [ x for x in range(pedInfo.nPeople) if inputGenotype[x] >= 0  ]
#
#	for i in genotypeIndex:
#		queryVector[i] = inputGenotype[i]

#	filename = "test/output_" + name + "_" + str(threading.get_ident()) + ".txt"

#	with open(filename, 'w') as f:
#		print(name, file=f)
#		for i in genotypeMatchIndex:
#			print(genotypeStates[i], " - ", genotypeProbabilities[i], file=f)
#		print("\n\n", file=f)

	# count number of transmissions and non-transmissions to get observed probability
	observedProbability = 0.0

	# if the proband is a non-carrier, conditional probability is zero
	if inputGenotype[proIndex] != 1:
		observedProbability = 0.0
	
	else:
		observedProbability = np.sum(genotypeProbabilities[genotypeMatchIndex])
		

#		# identify genotype founders
#		founderGenotypes = findGenotypeFounders(inputGenotype, pedInfo)
#		founderProbabilities = []
#
#
#		# loop over independent founders and calculate probability
#		for founder in founderGenotypes:
#
#			observedProbability = 1.0
#
#			# make a copy so as not to modify original
#			vector = inputGenotype.copy()
#			
#			# make the founder a carrier if possible
#			if vector[founder] < 0:
#				vector[founder] = 1
#
#			# identify all carriers of the original variant
#			carriersIndex = [ x for x in range(pedInfo.nPeople) if inputGenotype[x] > 0 ]
#			
#
#			# if an individual is a descendant of the founder and an ancestor of a
#			# carrier, make them a carrier. Ignore if genotype is non-missing
#			for i in range(pedInfo.nPeople):
#				if vector[i] >= 0:
#					continue
#				elif pedInfo.descendantTable[i, founder]:
#					ances = False
#					for j in carriersIndex:
#						if pedInfo.descendantTable[j, i]:
#							ances = True
#					if ances:
#						vector[i] = 1
#
#
#			for i in range(pedInfo.nPeople):
#				if pedInfo.phenotypeActual[i] >= 0 and pedInfo.hasParents[i] and (vector[pedInfo.dadIndex[i]] == 1 or vector[pedInfo.mamIndex[i]] == 1) and vector[i] >= 0:
#					observedProbability = observedProbability / 2
#	
#			if observedProbability == 1.0:
#				observedProbability = 0.0
#
#			founderProbabilities.append(observedProbability)
#	
#		#print(geno(inputGenotype), " : ", founderGenotypes, " - ", founderProbabilities)
#
#		if len(founderProbabilities) > 0:
#			observedProbability = sum(founderProbabilities)/len(founderProbabilities)
#		else:
#			observedProbability = 0.0

		#print(name, " - ", genotypeString(inputGenotype), " - ", founder)

#	observedProbability = 0.0
#	for founder in proFounderIndex:
#		desc = True
#		vector = queryVector.copy()
#		for i in genotypeIndex:
#			if pedInfo.descendantTable[i, founder] == 0 and i != founder and pedInfo.hasParents[i]:
#				desc = False
#		if desc:
#			vector[founder] = 1.0
#		else:
#			continue
#		
#		for i in genotypeIndex:
#			for j in [x for x in range(pedInfo.nPeople) if pedInfo.descendantTable[i,x] == 1 ]:
#				if pedInfo.descendantTable[j,founder] == 1:
#					vector[j] = 1.0
#		genotypeMatchIndexTmp = [ x for x in range(len(genotypeStates)) if matchVectors(vector, genotypeStates[x]) ]
#		print(founder, "\t", genotypeMatchIndexTmp)
#
#		for i in range(pedInfo.nPeople):
#			print(i, "\t", pedInfo.indID[i], "\t", genotypeStates[genotypeMatchIndexTmp[0]][i])
#
#		observedProbability = observedProbability + genotypeProbabilities[genotypeMatchIndexTmp].sum()
		


	numerator = 0.0
	for index in genotypeMatchIndex:
		p = 1.0


		# set the penetrance and phenocopy rate by counting the number of occurences of the
		# variant (or reference allele) in the cases and controls 
		penetranceCount = 0
		notPenetranceCount = 0
		phenocopyCount = 0
		notPhenocopyCount = 0

		for i in range(pedInfo.nPeople):
			if pedInfo.phenotypeActual[i] == 1 and genotypeStates[index][i] == 1:
				penetranceCount += 1
			elif pedInfo.phenotypeActual[i] == 0 and genotypeStates[index][i] == 1:
				notPenetranceCount += 1
			elif pedInfo.phenotypeActual[i] == 1 and genotypeStates[index][i] == 0:
				phenocopyCount += 1
			else:
				notPhenocopyCount += 1


		phenotypeProbability[1,1] = float(penetranceCount) / float(penetranceCount + notPenetranceCount)
		if phenotypeProbability[1,1] == 1.0:
			phenotypeProbability[1,1] -= epsilon
		elif phenotypeProbability[1,1] == 0.0:
			phenotypeProbability[1,1] = epsilon
		phenotypeProbability[0,1] = 1-phenotypeProbability[1,1]

		phenotypeProbability[1,0] = float(phenocopyCount) / float(phenocopyCount + notPhenocopyCount)
		if phenotypeProbability[1,0] == 1.0:
			phenotypeProbability[1,0] -= epsilon
		elif phenotypeProbability[1,0] == 0.0:
			phenotypeProbability[1,0] = epsilon
		phenotypeProbability[0,0] = 1-phenotypeProbability[1,0]


		for i in range(pedInfo.nPeople):
			p = p*phenotypeProbability[pedInfo.phenotypeActual[i],genotypeStates[index][i]]
		numerator = numerator + p*genotypeProbabilities[index]
	

	denominator = 0.0
	for index in range(len(genotypeStates)):
		p = 1.0

		# set the penetrance and phenocopy rate by counting the number of occurences of the
		# variant (or reference allele) in the cases and controls 
		penetranceCount = 0
		notPenetranceCount = 0
		phenocopyCount = 0
		notPhenocopyCount = 0

		for i in range(pedInfo.nPeople):
			if pedInfo.phenotypeActual[i] == 1 and genotypeStates[index][i] == 1:
				penetranceCount += 1
			elif pedInfo.phenotypeActual[i] == 0 and genotypeStates[index][i] == 1:
				notPenetranceCount += 1
			elif pedInfo.phenotypeActual[i] == 1 and genotypeStates[index][i] == 0:
				phenocopyCount += 1
			else:
				notPhenocopyCount += 1


		phenotypeProbability[1,1] = float(penetranceCount) / float(penetranceCount + notPenetranceCount)
		if phenotypeProbability[1,1] == 1.0:
			phenotypeProbability[1,1] -= epsilon
		elif phenotypeProbability[1,1] == 0.0:
			phenotypeProbability[1,1] = epsilon
		phenotypeProbability[0,1] = 1-phenotypeProbability[1,1]

		phenotypeProbability[1,0] = float(phenocopyCount) / float(phenocopyCount + notPhenocopyCount)
		if phenotypeProbability[1,0] == 1.0:
			phenotypeProbability[1,0] -= epsilon
		elif phenotypeProbability[1,0] == 0.0:
			phenotypeProbability[1,0] = epsilon
		phenotypeProbability[0,0] = 1-phenotypeProbability[1,0]

		for i in range(pedInfo.nPeople):
			p = p*phenotypeProbability[pedInfo.phenotypeActual[i],genotypeStates[index][i]]
		denominator = denominator + p*genotypeProbabilities[index]


	#print("numerator:   ", numerator)
	#print("denominator: ", denominator)
	#print("\n")

	#print("P_d(P|G):    ", numerator/denominator)
	#print("P_n(P|G):    ", observedProbability)
	#print("\n")



	if denominator == 0 or observedProbability == 0:
		LR = float("inf")
	else:
		LR = numerator/(observedProbability*denominator)

	
	allLR[name] = [ LR, numerator/denominator, observedProbability ]

	return LR






# main function
def main(argv):


	# command line arguments
	inputFamFile = None
	inputVcfFile = None
	proID = None
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

		if opt in ("-p", "--proband"):
			proID = arg

		if opt in ("-v", "--vcf"):
			inputVcfFile = arg
	
		if opt in ("--prob1"):
			p1 = float(arg)

		if opt in ("--prob2"):
			p2 = float(arg)

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
	phenotypeProbability = np.zeros((2,2))
	phenotypeProbability[1,0] = p1
	phenotypeProbability[1,1] = p2
	phenotypeProbability[0,0] = 1-phenotypeProbability[1,0]
	phenotypeProbability[0,1] = 1-phenotypeProbability[1,1]
	






	################################################################################
	# set pedigree variables
	################################################################################

	logging.info("Setting pedigre variables")


	# get proband information
	if proID is None:
		proID = indID[nonFounderIndex[0]]

	try:
		proIndex = np.where(indID == proID)[0][0]

	except IndexError:
		logging.error("Specified proband ID is not found in input PED file.")
		sys.exit("Exiting ... ")







	################################################################################
	# set genotype information
	################################################################################

	logging.info("Finding genotype states")



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

	#genotypeActual = pedigreeFile[:,6].astype(np.int)
	#genotypes = np.transpose(pedigreeFile[:,6:].astype(np.int))
	genotypes = np.full((pedInfo.nPeople, nVariants), -9)


	# create list to hold inuqie variant ID
	varID = []


	# loop over all samples in VCF and get genotype
	j = 0
	for variant in vcf:
		# get the ID of the variant
		varID.append(variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0])


		#print(variant.gt_types, "\t", variant.genotypes)
		# fill the known genotypes
		for i in range(len(vcfSampleIndex)):

			# get genotype type: {0,1,2,3}
			gt = int(variant.gt_types[i])

			# dominant inheritance: HET and HOM_ALT are the same. 
			# missing genotypes are set to -9
			if gt == 2:
				gt = 1

			if gt == 3:
				gt = -9

			# set known genotype
			genotypes[vcfSampleIndex[i]][j] = gt
		j += 1

	varString = np.apply_along_axis(genotypeString, 0, genotypes)

	# transpose array for parallelisation
	genotypes = np.transpose(genotypes.astype(np.int))


	# create dictionary to store all LR
	manager = Manager()
	allLR = manager.dict()


	# partial function for parallelisation - all constant except the input genotypes
	func = partial(calculateLR, pedInfo, proIndex, phenotypeProbability, allLR)



#	for i in range(pedInfo.nPeople):
#		print(i, "\t", pedInfo.hasParents[i])

#	for i in range(len(pedInfo.descendantTable)):
#		print(pedInfo.indID[i], "\t", i, "\t", pedInfo.descendantTable[i])


	pool = Pool(nCores)

	results = pool.map(func, genotypes)

	pool.close()


#	for i in range(len(genotypes[:,0])):
#		LR = calculateLR(genotypes[i,:], pedInfo, proIndex, phenotypeProbability, allLR)


	################################################################################
	# Output
	################################################################################

	logging.info("Output")

	
	#for i in range(len(varID)):
		#print(varID[i], "\t", results[i], "\t", varString[i], "\t", varString[i].count("."))
		#print(results[i], "\t", varString[i], "\t", genotypes[i])

	pprint.pprint(dict(allLR))

	#print(genotypes)


	#print(findGenotypeFounders(genotypes[1], pedInfo))




if __name__ == "__main__":
	main(sys.argv[1:])

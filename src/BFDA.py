import BF_Functions
import csv
import itertools
import logging
import multiprocessing
import pygad
import random
import re
import sys

import numpy as np
import scipy.special as sp

from BF_Functions import *
from cyvcf2 import VCF
from multiprocessing import cpu_count, Pool, Manager
from functools import partial









# main function
def BFDA_main(args):


	logging.info("BAYES FACTOR DESIGN ANALYSIS")
	logging.info(" ")


	# command line arguments
	nCores = 1
	inputFamFile = None
	outputPrefix = None
	nSim = 10000
	global priorCaus
	priorCaus = "linear"
	global priorNeut
	priorNeut = "uniform"





	if args.cores is not None:
		if int(args.cores) <= cpu_count():
			nCores = int(args.cores)

	if args.fam is not None:
		inputFamFile = args.fam

	if args.prefix is not None:
		outputPrefix = args.prefix

	if args.simulations is not None:
		nSim = args.simulations

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
	global pedInfo
	pedInfo = Pedigree(np.unique(famID), indID, dadID, mamID, sexID, pheID)







	################################################################################
	# set global variables
	################################################################################

	#global binomCoeff
	BF_Functions.binomCoeff = [ [0]*(pedInfo.nPeople + 20) for _ in range(pedInfo.nPeople + 20) ]

	for i in range(pedInfo.nPeople + 20):
		for j in range(i+1):
			BF_Functions.binomCoeff[i][j] = float(sp.binom(i,j))




	################################################################################
	# set up for BFDA
	################################################################################

	# set up lock
	l = multiprocessing.Lock()
	lock_init(l)


	# create dictionary to store all BF
	manager = Manager()
	global allBF
	allBF = manager.dict()


	# set BF to zero for any variant with a HOM_ALT carrier
	allBF["HOM_ALT"] = [ 0.0, 0.0, 0.0, 0 ]


	# generate causal distribution
	if priorCaus == "uniform":
		x_beta = 1
		y_phi = 1
	
	elif priorCaus == "linear": 
		x_beta = 2
		y_phi = 2
	
	else
		logging.error("Prior distribution for parameters under causal model not known. ")
		
	
	causalGenotypes = []
	for i in range(nSim):
		beta = ( random.uniform(0,1) ) ** (1/x_beta)
		phi = 1 - ( 1 - random.uniform(0,1)*( 1 - (1-beta)**y_phi ) )**(1/y_phi)

		p_carr_case = beta / (beta + phi)
		p_carr_con = (1 - beta) / (2 - beta - phi)

		genotype = np.full(pedInfo.nPeople, -1)
		completed = np.full(pedInfo.nPeople, 0)

		founder = random.randrage(pedInfo.nPeople):
		genotype[founder] = 1
		completed[founder] = 1


		# non-descendants of the founder are non-carriers
		for i in range(pedInfo.nPeople):
			if pedInfo.descendantTable[i, founder] == 0:
				genotype[i] = 0
				completed[i] = 1

		oldCount = 0
		newCount = np.count_nonzero(genotype == -1)
		counter = 0


		while oldCount != newCount:
			completedParents = [ j for i in pedInfo.nonFounderIndex if completed[pedInfo.dadIndex[j]] and completed[pedInfo.mamIndex[j]] and completed[j] = 0 ]

			for child in completedParents:
				
				# if both parents are non-carriers, child is a non-carrier
				if genotype[pedInfo.dadIndex[child]] == 0 and genotype[pedInfo.mamIndex[child]] == 0:
					genotype[child] = 0
					completed[child] = 1

				# otherwise, inherit according to phenotype
				else:

					# carrier probabilities for cases
					if pedInfo.actualPhenotypes[child] == 1:
						if random.uniform(0,1) < p_carr_case:
							genotype[child] = 1
						else:
							genotype[child] = 0

					# carrier probabilities for controls
					else:
						if random.uniform(0,1) < p_carr_con:
							genotype[child] = 1
						else:
							genotype[child] = 0

					completed[child] = 1

			oldCount = newCount
			newCount = np.count_nonzero(genotype == -1)
			counter = counter + 1
				

		causalGenotypes.append(genotype)

	
	print(causalGenotypes)



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
		




	################################################################################
	# Output
	################################################################################

	logging.info("Output")


	if args.complete:
		index_min = max(range(len(BFs)), key=BFs.__getitem__)
		logging.info(f"Maximum BF is {np.log10(BFs[index_min])}")

		with open(args.outputDir + outputPrefix + ".BF.txt", 'w') as f:
			print("i\tBF\tlogBF\tSELECTED\tSTRING", file=f)
			for i in range(len(BFs)):
				print(i, "\t", BFs[i], "\t", np.log10(float(BFs[i])), "\t", selectedIDs[i], "\t", varString[i], file=f, sep="")


	else:
		with open(args.outputDir + outputPrefix + ".SelectSamples.txt", 'w') as f:
			print("BF\tlogBF\tN\tSAMPLES", file=f)
			print(solution_fitness, "\t", np.log10(solution_fitness), "\t", len(optimalSolutionID),"\t", np.array2string(optimalSolutionID, separator=',')[1:-1].replace(" ", "").replace("'", "").replace("\n", ""), file=f)

	

	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")





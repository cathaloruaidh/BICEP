import BF_Functions
import csv
import logging
import multiprocessing
import pygad
import re
import sys

import numpy as np

from BF_Functions import *
from cyvcf2 import VCF
from multiprocessing import cpu_count, Pool, Manager
from functools import partial




# global variables for pygad parallelisation
pedInfo = None
allBF = None
priorCaus = None
priorNeut = None


# fitness function
def fitness_func( ga_instance, solution, solution_idx ):
	global pedInfo
	global allBF
	global priorCaus
	global priorNeut
	global requiredIDX
	global selectedGreedy

	genotypes = np.full(pedInfo.nPeople, -1)


	for i in [ int(_) for _ in np.concatenate([solution, requiredIDX, selectedGreedy]) ]:
		genotypes[i] = pedInfo.phenotypeActual[i] 


	# set control obligate carriers to carriers
	for i in range(len(genotypes)):
		if genotypes[i] == 0:
			if pedInfo.hasParents[i] and ( genotypes[pedInfo.dadIndex[i]] + genotypes[pedInfo.mamIndex[i]] > 0):
				c = sum([ genotypes[child] for child in pedInfo.children[i] if genotypes[child] > 0 ])
				if c > 0:
					genotypes[i] = 1

	return calculateBF(pedInfo, allBF, [priorCaus, priorNeut], [genotypes, genotypeString(genotypes)])


# messaging each generation
def on_gen(ga_instance):
	msg = "Generation : " + str(ga_instance.generations_completed)
	logging.info(msg)

	msg = "Fitness of the best solution :" + str(ga_instance.best_solution()[1])
	logging.info(msg)





# main function
def SS_main(args):


	logging.info("SELECT SAMPLES")
	logging.info(" ")


	# command line arguments
	nCores = 1
	inputFamFile = None
	outputPrefix = None
	nSelected = 5
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

	if args.select is not None:
		nSelected = args.select

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



	# get list of samples available for selection
	if pedigreeFile.shape[1] == 7:
		availIDX = [i for i in range(pedInfo.nPeople) if int(pedigreeFile[i,6]) == 1]
	
	else:
		availIDX = range(pedInfo.nPeople)



	# get list of required samples
	global requiredIDX
	if pedigreeFile.shape[1] == 8:
		requiredIDX = [i for i in range(pedInfo.nPeople) if int(pedigreeFile[i,7]) == 1]
	
	else:
		requiredIDX = []


	# remove required from available, and adjust nSelected accordingly
	availIDX = [i for i in availIDX if i not in requiredIDX]

	if len(availIDX) < nSelected:
		logging.info(f"Note: {nSelected} individuals to be selected, but there are {len(availIDX)} individuals available, after including {len(requiredIDX)} required individuals. ")
		nSelected = len(availIDX)


	# greedy algorithm to select the cases which results in the highest number of obligate
	# carriers. This will help reduce the numebr of unknown genotypes. 
	global selectedGreedy
	selectedGreedy = []

	if args.greedy:
		logging.info("Greedy initial selection of cases. ")

		cases = [ i for i in availIDX if pedInfo.phenotypeActual[i] == 1 ]
		nonParentCases = [ i for i in cases if pedInfo.isParent[i] == False  ]

		carriers = np.full(pedInfo.nPeople, -1)
		for i in nonParentCases:
			carriers[i] = 1


		foundersID = getMRCA(carriers, pedInfo).split("|")
		foundersIDX = [ np.where(pedInfo.indID == founder) for founder in foundersID ]

		founderPick = foundersIDX[0]


		# for all nonParentCases, make a dictionary of their ancestors to the founder
		ancestors = dict((k, []) for k in nonParentCases)

		for case in nonParentCases:
			obligateParent = pedInfo.dadIndex[case] if pedInfo.descendantTable[pedInfo.dadIndex[case], founderPick] else pedInfo.mamIndex[case]
			
			# now iterate over all their parents until we hit the founder
			while obligateParent != founderPick:
				ancestors[case].append(obligateParent)
				obligateParent = pedInfo.dadIndex[obligateParent] if pedInfo.descendantTable[pedInfo.dadIndex[obligateParent], founderPick] else pedInfo.mamIndex[obligateParent]


		# add case who is most distant to founder, then remove all their obligate carriers
		# repeat until we have covered all obligate carriers, or until we have hit nSelected
		while len(selectedGreedy) < nSelected and len(ancestors) > 0:
			dist = 0
			keep = None
			for ind in ancestors.keys():
				if len(ancestors[ind]) > dist:
					dist = len(ancestors[ind])
					keep = ind
			if dist == 0:
				break

			removal = ancestors[keep]
	
			selectedGreedy.append(keep)

			# add person with highest number of non-covered obligate carriers
			for i in removal:
				for key, value in ancestors.items():
					if i in value:
						val2 = [ _ for _ in value if _ != i ]
						ancestors[key] = val2


			#print("Loop over removal:")
			#for i in removal:
			#	for k in ancestors.keys():
			#		if i in ancestors[k]:
			#			print(f"Removing {pedInfo.indID[i]} from ancestors of {pedInfo.indID[k]}")
			#			ancestors[k].remove(i)


			ancestors = {k: v for k, v in ancestors.items() if v}

		print([ pedInfo.indID[i] for i in selectedGreedy])

	# remove greedy selection from available, and adjust the nSelected accordingly
	availIDX = [i for i in availIDX if i not in selectedGreedy]
	nSelected = nSelected - len(selectedGreedy)


	


	################################################################################
	# set global variables
	################################################################################

	#global binomCoeff
	BF_Functions.binomCoeff = [ [0]*(pedInfo.nPeople + 20) for _ in range(pedInfo.nPeople + 20) ]

	for i in range(pedInfo.nPeople + 20):
		for j in range(i+1):
			BF_Functions.binomCoeff[i][j] = float(sp.binom(i,j))




	################################################################################
	# set up for genetic algorithm
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



	logging.info("Running genetic algorithm")

    # set the parameters of the GA
	num_generations = 50
	sol_per_pop = 500
	num_genes = nSelected
	gene_type=int
	gene_space = availIDX

	num_parents_mating = 50
	parent_selection_type = "sss"
	keep_parents = 50
	keep_elitism = 50
	crossover_type = "scattered"

	mutation_type = "random"
	mutation_num_genes = np.min([2, nSelected])

	parallel_processing = [ "process", nCores ]
	stop_criteria = ["saturate_10"]


	# initiate the GA to find optimal samples
	if nSelected > 0:
		ga_instance = pygad.GA(num_generations=num_generations,
		on_generation=on_gen,
		parallel_processing=parallel_processing,
		num_parents_mating=num_parents_mating,
		fitness_func=fitness_func,
		sol_per_pop=sol_per_pop,
		num_genes=num_genes,
		gene_type=gene_type,
		allow_duplicate_genes=False,
		gene_space = gene_space,
		parent_selection_type=parent_selection_type,
		keep_elitism=keep_elitism,
		crossover_type=crossover_type,
		mutation_type=mutation_type,
		mutation_num_genes=mutation_num_genes,
		stop_criteria=stop_criteria,
		suppress_warnings=True)

		ga_instance.run()
		solution, solution_fitness, solution_idx = ga_instance.best_solution()
	else:
		logging.info("Skipping genetic algorithm due to greedy selection")
		solution = []
		solution_fitness = fitness_func(None, solution, None)

	optimalSolutionID = np.array([ pedInfo.indID[int(i)] for i in np.concatenate([solution, requiredIDX, selectedGreedy]) ])
		



	################################################################################
	# Output
	################################################################################

	logging.info("Output")


	print("Parameters of the best solution : {solution}".format(solution=solution))
	print("Predicted output based on the best solution : {prediction}".format(prediction=np.log10(solution_fitness)))


	with open(args.outputDir + outputPrefix + ".SelectSamples.txt", 'w') as f:
		print("BF\tlogBF\tN\tSAMPLES", file=f)
		print(solution_fitness, "\t", np.log10(solution_fitness), "\t", len(optimalSolutionID),"\t", np.array2string(optimalSolutionID, separator=',')[1:-1].replace(" ", "").replace("'", "").replace("\n", ""), file=f)

	

	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")





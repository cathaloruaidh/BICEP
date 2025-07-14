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


# main function
def SS_main(args):


	logging.info("SELECT SAMPLES")
	logging.info(" ")


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

	if args.prefix is not None:
		outputPrefix = args.prefix

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



	# get list of samples available for selection
	if pedigreeFile.shape[1] == 7:
		availID = [indID[i] for i in range(pedIndo.nPeople) if pedigreeFile[i,6] == 1]
	
	else:
		availID = indID


	


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



	# create dictionary to store all BF
	manager = Manager()
	allBF = manager.dict()


	# set BF to zero for any variant with a HOM_ALT carrier
	allBF["HOM_ALT"] = [ 0.0, 0.0, 0.0, 0 ]


	# fitness function
        def fitness_func( ga_instance, solution, solution_idx ):
            nonlocal pedInfo
            nonlocal allBF
            nonlocal priorCaus
            nonlocal priorNeut

            genotypes = np.full(pedInfo.nPeople, -1)


            for i in solution:
                genotypes[i] = 1 if pedInfo.phenotypeActual[i] == 1 else 0

            return calculateBF(pedInfo, allBF, [priorCaus, priorNeut], [genotypes, genotypeString(genotypes)])


	logging.info("Running genetic algorithm")

        # set the parameters of the GA

        num_generations = 100
        sol_per_pop = 100
        num_genes = args.select
        gene_type=int
        init_range_low = 0
        init_range_high = pedInfo.nPeople 

        num_parents_mating = 10
        parent_selection_type = "sss"
        keep_parents = 10
        crossover_type = "scattered"

        mutation_type = "random"
        mutation_percent_genes = 100 / num_genes
        mutation_num_genes = 1
        random_mutation_min_val = 0.0
        random_mutation_max_val = 1.0



	# initiate the GA to find optimal samples
        ga_instance = pygad.GA(num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_type=gene_type,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_num_genes=mutation_num_genes)

        ga_instance.run()



	################################################################################
	# Output
	################################################################################

	logging.info("Output")

	

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        prediction = fitness_func(None, solution, None)
        print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

        

	

	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")





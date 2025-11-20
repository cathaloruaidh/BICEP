import BF_Functions
import csv
import logging
import multiprocessing
import re
import sys

import os
import psutil

import numpy as np

from BF_Functions import *
from cyvcf2 import VCF
from multiprocessing import cpu_count, Pool, Manager
from functools import partial


# main function
def BF_main(args):


	logging.info("BAYES FACTOR")
	logging.info(" ")
	mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
	print(f"Memory = {mem}")


	# command line arguments
	nCores = 1
	inputFamFile = None
	inputVcfFile = None
	outputPrefix = None
	outputLog = None
	minAffecteds = 0
	priorCaus = "linear"
	priorNeut = "uniform"
	branch = None





	if args.cores is not None:
		if int(args.cores) <= cpu_count():
			nCores = int(args.cores)

	if args.fam is not None:
		inputFamFile = args.fam

	#if args.minAff is not None:
	#	minAffecteds = int(args.minAff)

	if args.prefix is not None:
		outputPrefix = args.prefix

	if args.vcf is not None:
		inputVcfFile = args.vcf

	if args.priorCaus is not None:
		priorCaus = args.priorCaus

	if args.priorNeut is not None:
		priorNeut = args.priorNeut

	if args.branch is not None:
		branch = args.branch




	# up recursion limit
	sys.setrecursionlimit(10000)



	# read contents of file into np array
	logging.info("Reading input FAM file")
	mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
	print(f"Memory = {mem}")


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


	# print descendant table to file
	with open(args.tempDir + args.prefix + '.descendantTable.txt', 'w') as f:
		print(','.join(map(str, np.concatenate([np.array(["#"]), pedInfo.indID]))), file = f)
	
	dt_df = pd.DataFrame(pedInfo.descendantTable, columns=pedInfo.indID, index=pedInfo.indID)
	dt_df.to_csv(args.tempDir + args.prefix + '.descendantTable.txt', mode='a', header=False)

	
	# print relationship matrix to file
	with open(args.tempDir + args.prefix + '.kinshipMatrix.txt', 'w') as f:
		print(','.join(map(str, np.concatenate([np.array(["#"]), pedInfo.indID]))), file = f)
	
	km_df = pd.DataFrame(pedInfo.kinshipMatrix, columns=pedInfo.indID, index=pedInfo.indID)
	km_df.to_csv(args.tempDir + args.prefix + '.kinshipMatrix.txt', mode='a', header=False)


	# identify the main branch of the family

	


	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")
	mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
	print(f"Memory = {mem}")




	# get VCF file and variant/sample information
	vcf = VCF(inputVcfFile, gts012=True)
	
	
	# check if file is binary
	with open(inputVcfFile, 'rb') as test_f:
		if test_f.read(2) == b'\x1f\x8b':
			nVariants = sum(1 for line in gzip.open(inputVcfFile, mode='r') if not bool(re.match("^#", line.decode('utf-8'))))
		else:
			nVariants = sum(1 for line in open(inputVcfFile) if not bool(re.match("^#", line)))
	
	

	vcfSampleIndex = []

	for i in range(len(vcf.samples)):
		try:
			ind = np.where(pedInfo.indID == vcf.samples[i])[0][0]
		except:
			msg = "Sample " + vcf.samples[i] + " is in VCF but not FAM."
			logging.critical(msg)
		else:
			vcfSampleIndex.append(ind)


	logging.info("Store as np array")
	mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
	print(f"Memory = {mem}")


	# set all genotypes to missing as input
	genotypes = np.full((pedInfo.nPeople, nVariants), -1)
	homozygous = np.full(nVariants, False)


	# create list to hold unique variant ID
	varID = []


	# get IDs of variants which recieved a prior
	#prior_IDs = None
	#with open(args.outputDir + args.prefix + '.priors.txt', 'rb') as f:
	#	 prior_IDs = pd.read_csv(f, dtype=str, na_values = ['.'], sep = '\t')
	#	 prior_IDs = prior_IDs.dropna(subset=['logPriorOC'])
	#	 prior_IDs = prior_IDs["ID"]


	# loop over all samples in VCF and get genotype
	j = 0
	CHROMS = set([ "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22" ])

	GT_dict = { "0/0" : 0, "0/1" : 1, "1/0" : 1, "1/1" : 2, "./." : 3  }

	for variant in vcf:
		# remove variants not on autosomes
		if len( set([variant.CHROM, "chr"+variant.CHROM]) & CHROMS ) == 0:
			continue

		# get the ID of the variant
		if args.cnv:
			#var_tmp = variant.CHROM + "_" + str(variant.start+1) + "_" + str(variant.INFO.get('END')) + "_" + variant.INFO.get('SVTYPE')
			var_tmp = variant.ID

		else:
			var_tmp = variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0]
		varID.append(var_tmp)

		#if prior_IDs is not None:
		#	if prior_IDs.str.contains(var_tmp).any():
		#		varID.append(var_tmp)
		#	else:
		#		continue

	
		if args.key:
			gt_list = variant.format(args.key)

			for i in range(len(gt_list)):
				if "/" in gt_list[i]:
					gt_list[i] = GT_dict[gt_list[i]]

				if gt_list[i] == ".":
					gt_list[i] = 3

		else:
			gt_list = variant.gt_types



		# fill the known genotypes
		for i in range(len(vcfSampleIndex)):

			# get genotype type: {0,1,2,3}
			gt = int(gt_list[i])

			# dominant inheritance: HET and HOM_ALT are the same. 
			# missing genotypes are set to -1
			if gt == 2:
				gt = 1
				homozygous[j] = True

			if gt == 3:
				gt = -1

			# set known genotype
			genotypes[vcfSampleIndex[i]][j] = gt
		j += 1

	# ignore variants which haven't recieved a prior
	#if prior_IDs is not None:
	#	logging.info("Remove variants with no prior")
	#	ind = np.in1d(varID, prior_IDs)
	#	varID = varID[ind]
	#	genotypes = genotypes[:][ind]


	varString = np.apply_along_axis(genotypeString, 0, genotypes)


	# for the variants with a HOM_ALT carrier, change the
	# variant string so that a BF won't be calculated
	np.putmask(varString, homozygous, "HOM_ALT")


	# transpose array for parallelisation
	genotypes = np.transpose(genotypes.astype(int))


	# combine variant name with genotypes
	data = [ (genotypes[i],varString[i]) for i in range(len(genotypes)) ]


	################################################################################
	# set global variables
	################################################################################

	#global binomCoeff
	BF_Functions.binomCoeff = [ [0]*(pedInfo.nPeople + 20) for _ in range(pedInfo.nPeople + 20) ]

	for i in range(pedInfo.nPeople + 20):
		for j in range(i+1):
			BF_Functions.binomCoeff[i][j] = float(sp.binom(i,j))




	################################################################################
	# calculate Bayes factors
	################################################################################



	# create dictionary to store all BF
	manager = Manager()
	allBF = manager.dict()


	# set BF to zero for any variant with a HOM_ALT carrier
	allBF["HOM_ALT"] = [ 0.0, 0.0, 0.0, 0 ]


#	if minAffecteds > 0:
#		msg = "Removing variants with minAff < " + str(minAffecteds)
#		logging.info(msg)
#
#		for i in range(len(genotypes)):
#
#			count = 0
#			affs = [ x for x in range(pedInfo.nPeople) if pedInfo.phenotypeActual[x] == 1 ]
#
#			for aff in affs:
#				if genotypes[i][aff] == 1:
#					count += 1
#
#			if count < minAffecteds:
#				allBF[varString[i]] = [ 0.0, 0.0, 0.0, 0 ]
#
#				msg = "Removed variant: " + varString[i]
#				logging.debug(msg)



	logging.info("Calculating Bayes Factors")
	mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
	print(f"Memory = {mem}")


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
	mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
	print(f"Memory = {mem}")

	
	if outputPrefix is None:
		for i in range(len(varID)):
			print(varID[i], "\t", BFs[i], "\t", varString[i], "\t", sep="")
	else:
		with open(args.outputDir + outputPrefix + ".BF.txt", 'w') as f:
			print("ID\tBF\tlogBF\tSTRING\tAFF_CARR\tAFF_NON-CARR\tUNAFF_CARR\tUNAFF_NON-CARR\tMISS\tMRCA", file=f)
			for i in range(len(varID)):
				aff_c, aff_nc, un_c, un_nc, miss = phenoCarriers(genotypes[i], pedInfo, vcf.samples)
				MRCA = getMRCA(genotypes[i], pedInfo)
				print(varID[i], "\t", BFs[i], "\t", np.log10(float(BFs[i])), "\t", varString[i], "\t", aff_c, "\t", aff_nc, "\t", un_c, "\t", un_nc, "\t", miss, "\t", MRCA, file=f, sep="")
		


	# get the best co-segregation score
	maxBF, founder = getMaxBF(pedInfo, allBF, [priorCaus, priorNeut], vcfSampleIndex)

	with open(args.tempDir + outputPrefix + ".max_logBF.txt", 'w') as f:
		print(np.log10(float(maxBF)), file=f)
		print(f"{founder}", file=f)

	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")





import BF_Functions
import csv
import itertools
import logging
import multiprocessing
import pygad
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
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
	nBreaks = 1000
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

	if args.breaks is not None:
		nBreaks = args.breaks

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
	
	elif "," in priorCaus:
		if len(priorCaus.split(",")) != 2:
			logging.error(f"Error with Beta distribution parameters: {priorCaus}")

		x_beta, y_phi = [ int(c) for c in priorCaus.split(",") ]
	else:
		logging.error("Prior distribution for parameters under causal model not known. ")

	y_m = 230
		
	
	logging.info("Generating the causal distribution. ")
	causalGenotypes = []

	beta_sampled = []
	beta_observed = []

	phi_sampled = []
	phi_observed = []


	# calculate the mininum phenocopy rate  if an individual were a founder
	min_phenocopy = np.full(pedInfo.nPeople, 0.0)
	for x in range(pedInfo.nPeople):
		non_desc_cases = [ y for y in range(pedInfo.nPeople) if pedInfo.descendantTable[y,x] == 0 and pedInfo.phenotypeActual[y] == 1 ]
		min_phenocopy[x] = len(non_desc_cases) / np.sum(pedInfo.phenotypeActual)



	i = 0

	while i < nSim:
		beta = ( random.uniform(0,1) ) ** (1/x_beta)
		phi = 1 - ( 1 - random.uniform(0,1)*( 1 - (1-beta)**y_phi ) )**(1/y_phi)
		m = 1 - (( 1 - random.uniform(0,1) ) ** (1/y_m))

		p_carr_case = beta / (beta + phi)
		p_carr_con = (1 - beta) / (2 - beta - phi)

		p_found_case = beta*m / ( beta*m + phi*(1-m) )
		p_found_con = (1-beta)*m / ( 1 - beta*m - phi*(1-m) )

		prob_founder = [ p_found_case if pedInfo.phenotypeActual[j] ==  1 else p_found_con for j in range(pedInfo.nPeople) ]
		# if the phenocopy rate can never get low enough, exclude that individual
		prob_founder = np.where(min_phenocopy > phi, 0, prob_founder)
		prob_founder = prob_founder/np.sum(prob_founder)

		founder = np.random.choice(pedInfo.nPeople, 1, p=prob_founder)[0]

		genotype = np.full(pedInfo.nPeople, -1)
		completed = np.full(pedInfo.nPeople, 0)
	
		genotype[founder] = 1
		completed[founder] = 1


		# non-descendants of the founder are non-carriers
		for j in range(pedInfo.nPeople):
			if pedInfo.descendantTable[j, founder] == 0:
				genotype[j] = 0
				completed[j] = 1

		oldCount = 0
		newCount = np.count_nonzero(genotype == -1)
		counter = 0


		while oldCount != newCount:
			completedParents = [ j for j in pedInfo.nonFounderIndex if completed[pedInfo.dadIndex[j]] and completed[pedInfo.mamIndex[j]] and completed[j] == 0 ]

			for child in completedParents:
				
				# if both parents are non-carriers, child is a non-carrier
				if genotype[pedInfo.dadIndex[child]] == 0 and genotype[pedInfo.mamIndex[child]] == 0:
					genotype[child] = 0
					completed[child] = 1

				# otherwise, inherit according to phenotype
				else:

					# carrier probabilities for cases
					if pedInfo.phenotypeActual[child] == 1:
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
				

		# if there is only one carrier, or no affected carriers, ignore and restart loop
		aff_carr = [ j for j in range(pedInfo.nPeople) if genotype[j] == 1 and pedInfo.phenotypeActual[j] == 1 ]

		if np.sum(genotype) == 1 or len(aff_carr) == 0:
			continue

		aff_non_carr = [ j for j in range(pedInfo.nPeople) if genotype[j] == 0 and pedInfo.phenotypeActual[j] == 1 ]


		causalGenotypes.append(genotype)
		beta_sampled.append(beta)
		beta_observed.append(len(aff_carr) / np.sum(genotype))
		phi_sampled.append(phi)
		phi_observed.append( len(aff_non_carr) / ( pedInfo.nPeople - np.sum(genotype) ))

		i = i+1

	causalString = np.apply_along_axis(genotypeString, 1, causalGenotypes)




	# generate neutral distribution
	logging.info("Generating the neutral distribution. ")
	neutralGenotypes = []


	i = 0
	while i < nSim:

		p_carr_case = 0.5
		p_carr_con = 0.5

		genotype = np.full(pedInfo.nPeople, -1)
		completed = np.full(pedInfo.nPeople, 0)

		founder = random.randrange(pedInfo.nPeople)
		genotype[founder] = 1
		completed[founder] = 1


		# non-descendants of the founder are non-carriers
		for j in range(pedInfo.nPeople):
			if pedInfo.descendantTable[j, founder] == 0:
				genotype[j] = 0
				completed[j] = 1

		oldCount = 0
		newCount = np.count_nonzero(genotype == -1)
		counter = 0


		while oldCount != newCount:
			completedParents = [ j for j in pedInfo.nonFounderIndex if completed[pedInfo.dadIndex[j]] and completed[pedInfo.mamIndex[j]] and completed[j] == 0 ]

			for child in completedParents:
				
				# if both parents are non-carriers, child is a non-carrier
				if genotype[pedInfo.dadIndex[child]] == 0 and genotype[pedInfo.mamIndex[child]] == 0:
					genotype[child] = 0
					completed[child] = 1

				# otherwise, inherit according to phenotype
				else:

					# carrier probabilities for cases
					if pedInfo.phenotypeActual[child] == 1:
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
				

		# if there is only one carrier, or no affected carriers, ignore and restart loop
		aff_carr = [ j for j in range(pedInfo.nPeople) if genotype[j] == 1 and pedInfo.phenotypeActual[j] == 1 ]
		if np.sum(genotype) == 1 or len(aff_carr) == 0:
			continue

		neutralGenotypes.append(genotype)
		i = i+1

	

	neutralString = np.apply_along_axis(genotypeString, 1, neutralGenotypes)

	data = [ (causalGenotypes[i],causalString[i]) for i in range(nSim) ] + [ (neutralGenotypes[i],neutralString[i]) for i in range(nSim) ]

	dataString = np.append(causalString, neutralString)

	modelString = ['CAUSAL']*nSim + ['NEUTRAL']*nSim 
	beta_sampled = beta_sampled + [0]*nSim
	beta_observed = beta_observed + [0]*nSim
	phi_sampled = phi_sampled + [0]*nSim
	phi_observed = phi_observed + [0]*nSim





	# calculate the Bayes Factor for all variants
	logging.info("Calculating the Bayes factors")
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
		for i in range(2*nSim):
			BFs.append(calculateBF(pedInfo, allBF, [priorCaus, priorNeut], data[i]))
		

	# evaluate thresholds
	logging.info("Evaluating the logBF thresholds")

	#maxBF, founder = getMaxBF(pedInfo, allBF, ["linear", "uniform"])
	maxBF = max([ x for x in BFs if x > 0 ])
	minBF = min([ x for x in BFs if x > 0 ])

	# get scale for plotting colour map
	diff = np.log10(maxBF/minBF)
	scale = np.floor(np.log10(np.log10(maxBF/minBF)))

	for w in range(-5,5):
		upper = np.ceil(np.log10(maxBF) * (2**w)) / (2**w)
		lower = np.floor(np.log10(minBF) * (2**w)) / (2**w)
		nCol = (upper - lower) / (2**(-w))
		if nCol > 5 and nCol <= 10:
			break
	

	#scale = np.floor(np.log10(np.log10(maxBF/minBF)))
	#upper = np.ceil( np.log10(maxBF) * (10**scale) ) / (10**scale)
	#lower = np.floor( np.log10(minBF) * (10**scale) ) / (10**scale)
	#nCol = (upper - lower) * (10**scale)
	#
	## min of 6 colours, max of 10 colours
	#while nCol < 6:
	#	nCol = nCol*2

	#thresholds = np.linspace(0, np.log10(maxBF), nBreaks)
	thresholds = np.linspace(lower, upper, nBreaks+1)
	TPE = []
	FPE = []
	PREC = []

	data_df = pd.DataFrame({'logBF': np.log10(BFs), 'model': modelString})

	# get rid of any entries with logBF = -inf
	data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
	data_df.dropna(subset=["logBF"], how="all", inplace=True)


	data_df_causal_len = len(data_df[ data_df["model"] == "CAUSAL" ].index)
	data_df_neutral_len = len(data_df[ data_df["model"] == "NEUTRAL" ].index)

	for t in thresholds:
		m = data_df[ data_df["logBF"] >= t ]["model"].to_numpy()

		TPE.append( sum(m == 'CAUSAL') / data_df_causal_len )
		FPE.append( sum(m == 'NEUTRAL') / data_df_neutral_len )
		if len(m) == 0:
			PREC.append( np.nan )
		else:
			PREC.append( sum(m == 'CAUSAL') / len(m) )

	# ensure the values are sorted correctly
	ord = np.argsort(-thresholds)
	#within = [ i for i in range(nBreaks) if thresholds[i] <= np.log10(maxBF) and thresholds[i] >= np.log10(minBF) ]
	thresholds = [ thresholds[i] for i in ord ]
	TPE = [ TPE[i] for i in ord ]
	FPE = [ FPE[i] for i in ord ]
	PREC = [ PREC[i] for i in ord ]


	# get thresholds for minimum precision
	i_PREC_90 = max([i for i in range(len(PREC)) if PREC[i] > 0.9 ], default=math.nan)
	T_PREC_90 = thresholds[i_PREC_90] if 0 <= i_PREC_90 < len(thresholds) else math.nan
	TPE_PREC_90 = TPE[i_PREC_90] if 0 <= i_PREC_90 < len(TPE) else math.nan
	FPE_PREC_90 = FPE[i_PREC_90] if 0 <= i_PREC_90 < len(FPE) else math.nan

	i_PREC_95 = max([i for i in range(len(PREC)) if PREC[i] > 0.95 ], default=math.nan)
	T_PREC_95 = thresholds[i_PREC_95] if 0 <= i_PREC_95 < len(thresholds) else math.nan
	TPE_PREC_95 = TPE[i_PREC_95] if 0 <= i_PREC_95 < len(TPE) else math.nan
	FPE_PREC_95 = FPE[i_PREC_95] if 0 <= i_PREC_95 < len(FPE) else math.nan

	i_PREC_99 = max([i for i in range(len(PREC)) if PREC[i] > 0.99 ], default=math.nan)
	T_PREC_99 = thresholds[i_PREC_99] if 0 <= i_PREC_99 < len(thresholds) else math.nan
	TPE_PREC_99 = TPE[i_PREC_99] if 0 <= i_PREC_99 < len(TPE) else math.nan
	FPE_PREC_99 = FPE[i_PREC_99] if 0 <= i_PREC_99 < len(FPE) else math.nan

	cutoff_df = pd.DataFrame({'Prec.' : ['90%', '95%', '99%'],
	'Thresh.' : [T_PREC_90, T_PREC_95, T_PREC_99], 
	'TPE' : [TPE_PREC_90, TPE_PREC_95, TPE_PREC_99],
	'FPE' : [FPE_PREC_90, FPE_PREC_95, FPE_PREC_99]})
	names = cutoff_df.columns
	cutoff_df = cutoff_df.round(3).T



	# compute AUC by trapezoidal rule
	AUC_ROC = 0.0
	for i in range(len(thresholds)-1):
		AUC_ROC = AUC_ROC + (FPE[i+1] - FPE[i])*(TPE[i] + TPE[i+1])/2


	AUC_PR = 0.0
	PREC = np.nan_to_num(PREC, nan=1)
	for i in range(len(thresholds)-1):
		AUC_PR = AUC_PR + (TPE[i+1] - TPE[i])*(PREC[i] + PREC[i+1])/2


	################################################################################
	# Output
	################################################################################

	logging.info("Output")


	cmap = plt.get_cmap('rainbow', nCol)
	
	# plot ROC curve
	#plt.title(f"BFDA ROC Curve - {args.prefix}")
	#plt.plot(FPE, TPE, c='black', label = 'AUC = %0.2f' % AUC_ROC)
	#plt.scatter(FPE,TPE, c=thresholds, cmap=cmap)
	#cbar = plt.colorbar()
	#plt.clim(lower, upper)
	#cbar.ax.set_ylabel('logBF threshold', rotation=270)
	#cbar.ax.get_yaxis().labelpad = 15
	#plt.legend(loc = 'lower right')
	#plt.plot([0, 1], [0, 1],'r--')
	#plt.xlim([0, 1])
	#plt.ylim([0, 1])
	#plt.ylabel('True Positive Evidence')
	#plt.xlabel('False Positive Evidence')
	#plt.savefig(args.outputDir + args.prefix + ".BFDA.ROC.png", dpi=300)


	## plot precision-recall curve
	#plt.clf()
	#plt.title(f"BFDA Precision/Recall - {args.prefix}")
	#plt.plot(TPE, PREC, c='black', label = 'AUC = %0.2f' % AUC_PR)
	#plt.scatter(TPE, PREC, c=thresholds, cmap=cmap)
	#cbar = plt.colorbar()
	#plt.clim(lower, upper)
	#cbar.ax.set_ylabel('logBF threshold', rotation=270)
	#cbar.ax.get_yaxis().labelpad = 15
	#plt.plot([0, 1], [0.5, 0.5],'r--')
	#plt.legend(loc = 'lower right')
	#plt.xlim([0,1])
	#plt.ylim([0,1])
	#plt.xlabel('Recall')
	#plt.ylabel('Precision')
	#plt.savefig(args.outputDir + args.prefix + ".BFDA.PRC.png", dpi=300)


	# plot all metrics on the same graph
	#plt.clf()
	#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, layout="constrained")
#
	#ax1.margins(0.05, 0.2)
	#ax1.set_ylim([0,1])
	#ax1.set(ylabel="TPE")
	#ax1.axhline(y=0.8, linewidth=1, color='C3', linestyle='--')
	#ax1.axhline(y=0.9, linewidth=1, color='C4', linestyle='--')
	#ax1.axvline(x=np.log10(minBF), linewidth=1, color='k', linestyle=':')
	#ax1.axvline(x=np.log10(maxBF), linewidth=1, color='k', linestyle=':')
	#ax1.plot(thresholds,TPE, c='C0')
	#ax1.tick_params(labelsize=7, labelright=True, right=True)
	#ax1.grid(linewidth=0.5, axis="x")


	#ax2.margins(0.05, 0.2)
	#ax2.set_ylim([0,1])
	#ax2.set(ylabel="FPE")
	#ax2.axvline(x=np.log10(minBF), linewidth=1, color='k', linestyle=':')
	#ax2.axvline(x=np.log10(maxBF), linewidth=1, color='k', linestyle=':')
	#ax2.plot(thresholds,FPE, c='C1')
	#ax2.tick_params(labelsize=7, labelright=True, right=True)
	#ax2.grid(linewidth=0.5, axis="x")


	#ax3.margins(0.05, 0.2)
	#ax3.set_ylim([0,1])
	#ax3.set(xlabel="logBF threshold", ylabel="Precision")
	#ax3.axhline(y=0.8, linewidth=1, color='C3', linestyle='--')
	#ax3.axhline(y=0.9, linewidth=1, color='C4', linestyle='--')
	#ax3.axvline(x=np.log10(minBF), linewidth=1, color='k', linestyle=':')
	#ax3.axvline(x=np.log10(maxBF), linewidth=1, color='k', linestyle=':')
	#ax3.plot(thresholds,PREC, c='C2')
	#ax3.set_xticks(np.arange(lower, upper, (upper-lower)/nCol))
	#ax3.set_xticks(np.arange(lower, upper, (upper-lower)/(nCol*5)), minor = True)
	#ax3.tick_params(labelsize=7, labelright=True, right=True)
	#ax3.grid(linewidth=0.5, axis="x")

	#plt.savefig(args.outputDir + args.prefix + ".BFDA.ALL.png", dpi=300)



	# plot all metrics on the same graph
	plt.clf()
	plt.rcParams.update({'font.size': 12})
	fig = plt.figure(figsize=[8.3, 11.7])
	ax_dict = fig.subplot_mosaic("AA;BB;CD", width_ratios=[3,2])

	ax_dict["A"].text(-0.1, 1.1, "A", transform=ax_dict["A"].transAxes, size=20, weight='bold')
	ax_dict["A"].title.set_text(f"Precision plot")
	ax_dict["A"].plot(thresholds,PREC, c='C0', linewidth=3)
	ax_dict["A"].margins(0.05, 0.2)
	ax_dict["A"].set_ylim([0.4,1])
	ax_dict["A"].set(xlabel="logBF threshold", ylabel="Precision")
	if math.isnan(T_PREC_90):
		ax_dict["A"].axhline(y=0.9, linewidth=2, color='C6', linestyle='--')
	else:
		ax_dict["A"].hlines(y=0.9, xmin=lower, xmax=T_PREC_90, linewidth=2, color='C6', linestyles="dashed")
		ax_dict["A"].vlines(x=T_PREC_90, ymin=0.4, ymax=0.9, linewidth=2, color='C6', linestyles="dashed")

	if math.isnan(T_PREC_95):
		ax_dict["A"].axhline(y=0.95, linewidth=2, color='C3', linestyle='--')
	else:
		ax_dict["A"].hlines(y=0.95, xmin=lower, xmax=T_PREC_95, linewidth=2, color='C4', linestyles="dashed")
		ax_dict["A"].vlines(x=T_PREC_95, ymin=0.4, ymax=0.95, linewidth=2, color='C4', linestyles="dashed")

	if math.isnan(T_PREC_99):
		ax_dict["A"].axhline(y=0.99, linewidth=2, color='C3', linestyle='--')
	else:
		ax_dict["A"].hlines(y=0.99, xmin=lower, xmax=T_PREC_99, linewidth=2, color='C3', linestyles="dashed")
		ax_dict["A"].vlines(x=T_PREC_99, ymin=0.4, ymax=0.99, linewidth=2, color='C3', linestyles="dashed")

	ax_dict["A"].axvline(x=np.log10(minBF), linewidth=1, color='k', linestyle=':')
	ax_dict["A"].axvline(x=np.log10(maxBF), linewidth=1, color='k', linestyle=':')
	ax_dict["A"].set_xticks(np.arange(lower, upper, (upper-lower)/nCol))
	ax_dict["A"].set_xticks(np.arange(lower, upper, (upper-lower)/(nCol*5)), minor = True)
	ax_dict["A"].tick_params(labelright=True, right=True)
	ax_dict["A"].grid(linewidth=0.5, axis="x")
	

	ax_dict["B"].text(-0.1, 1.1, "B", transform=ax_dict["B"].transAxes, size=20, weight='bold')
	ax_dict["B"].title.set_text(f"Histogram")
	ax_dict["B"].set(xlabel="logBF threshold", ylabel="Count")
	ax_dict["B"].hist(data_df[ data_df["model"] == "CAUSAL" ]["logBF"].to_numpy(), bins=np.linspace(lower, upper, int(nCol*2)), alpha=0.7, label="Causal", color="C1")
	ax_dict["B"].hist(data_df[ data_df["model"] != "CAUSAL" ]["logBF"].to_numpy(), bins=np.linspace(lower, upper, int(nCol*2)), alpha=0.7, label="Neutral", color="C2")
	ax_dict["B"].legend(loc = 'upper center')
	ax_dict["B"].sharex(ax_dict["A"])


	ax_dict["C"].text(-0.1, 1.1, "C", transform=ax_dict["C"].transAxes, size=20, weight='bold')
	ax_dict["C"].title.set_text(f"ROC Curve")
	ax_dict["C"].plot(FPE, TPE, c='black', label = 'AUC = %0.2f' % AUC_ROC)
	pcb = ax_dict["C"].scatter(FPE,TPE, c=thresholds, cmap=cmap)
	ax_dict["C"].legend(loc = 'lower right')
	cbar = fig.colorbar(pcb, ax=ax_dict["C"])
	#fig.clim(lower, upper)
	cbar.ax.set_ylabel('logBF threshold', rotation=270)
	cbar.ax.set_ylim(lower,upper)
	cbar.ax.set_yticks(np.arange(lower, upper, (upper-lower)/nCol))
	cbar.ax.get_yaxis().labelpad = 15
	ax_dict["C"].plot([0, 1], [0, 1],'--', color='C5')
	ax_dict["C"].set_xlim([0, 1])
	ax_dict["C"].set_ylim([0, 1])
	ax_dict["C"].set_ylabel('True Positive Evidence')
	ax_dict["C"].set_xlabel('False Positive Evidence')


	ax_dict["D"].axis('off')
	ax_dict["D"].axis('tight')
	ax_dict["D"].text(-0.1, 1.1, "D", transform=ax_dict["D"].transAxes, size=20, weight='bold')
	ax_dict["D"].title.set_text(f"Metrics")
	tab = ax_dict["D"].table(cellText=cutoff_df.values, rowLabels=names, loc='center')
	tab.scale(xscale=1, yscale=3)
	

	fig.subplots_adjust(hspace=0.5, wspace=1)
	#fig.tight_layout()


	#fig.canvas.draw()
	#bbox = ax_dict["A"].get_tightbbox(fig.canvas.get_renderer())
	#fig.text(bbox.x0, bbox.y1, "A", fontsize=12, fontweight="bold", va="top", ha="left",transform=None)

	#bbox = ax_dict["B"].get_tightbbox(fig.canvas.get_renderer())
	#fig.text(bbox.x0, bbox.y1, "B", fontsize=12, fontweight="bold", va="top", ha="left",transform=None)

	#bbox = ax_dict["C"].get_tightbbox(fig.canvas.get_renderer())
	#fig.text(bbox.x0, bbox.y1, "C", fontsize=12, fontweight="bold", va="top", ha="left",transform=None)

	plt.savefig(args.outputDir + args.prefix + ".BFDA.NEW.png", dpi=600)



	with open(args.tempDir + outputPrefix + ".BFDA.raw.txt", 'w') as f:
		print("i\tBF\tlogBF\tSTRING\tMODEL\tbeta_sampled\tbeta_observed\tphi_sampled\tphi_observed", file=f)
		for i in range(len(BFs)):
			print(f"{i}\t{BFs[i]}\t{np.log10(float(BFs[i]))}\t{dataString[i]}\t{modelString[i]}\t{beta_sampled[i]}\t{beta_observed[i]}\t{phi_sampled[i]}\t{phi_observed[i]}", file=f, sep="")


	thresh_df = pd.DataFrame({'Threshold': thresholds, 'FPE': FPE, 'TPE': TPE, 'Precision': PREC})
	thresh_df.to_csv(args.tempDir + outputPrefix + ".BFDA.thresholds.txt", index=False, sep='\t', na_rep='.')
		
	

	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")





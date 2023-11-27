import csv
import logging
import math
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import BayesFactor

# main function
def PO_main(args):

	priorFile = None
	bfFile = None
	

	# command line arguments
	if (args.input is None) and (args.prior is None) and (args.bf is None):
		priorFile = args.prefix + ".priors.txt"
		bfFile = args.prefix + ".BF.txt"

	else:
		if args.input is not None:
			priorFile = args.input + ".priors.txt"
			bfFile = args.input + ".BF.txt"

		if args.prior is not None:
			priorFile = args.prior

		if args.bf is not None:
			bfFile = args.bf


	# load the prior and BF files
	logging.info("Reading in the prior file")
	with open(args.outputDir + priorFile) as f:
		prior = pd.read_csv(f, sep="\t")
	
	
	logging.info("Reading in the Bayes factor file")
	with open(args.outputDir + bfFile) as f:
		bf = pd.read_csv(f, sep="\t")
	


	# combine values and output
	logging.info("Merge and output")
	merged = prior.merge(bf, on="ID")
	merged["logPostOC"] = merged["logPriorOC"] + merged["logBF"]


	with open(args.tempDir + args.input + '.max_logBF.txt', 'r') as f:
		tmp = f.readlines()
		max_logBF = float(tmp[0])

	merged_sub = merged.sort_values(by=['logPostOC'], ascending=False).head(n=args.top)
	merged_sub["Rank"] = range(1, args.top + 1)

	merged.to_csv(args.outputDir + args.prefix + ".posteriors.txt", index=False, sep='\t', na_rep='.')




	# plot top variants
	logging.info("Plotting the top " + str(args.top) +  " variants")
	min_y = np.floor(np.min(np.concatenate((merged_sub['logPostOC'].values, merged_sub['logBF'].values, merged_sub['logPriorOC'].values, np.array([max_logBF])))))
	max_y = np.ceil(np.max(np.concatenate((merged_sub['logPostOC'].values, merged_sub['logBF'].values, merged_sub['logPriorOC'].values, np.array([max_logBF])))))


	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

	x_ticks = [1] + [ i for i in range(5, args.top + 1, 5) ]
	x_ticks_label = [ str(i) for i in x_ticks ]

	plt.setp((ax1, ax2, ax3), xticks=x_ticks, xticklabels=x_ticks_label, ylim=(min_y, max_y))
	plt.gca().set_ylim(min_y, max_y)

	if (args.highlight is not None) and (merged_sub["ID"].str.contains(args.highlight)):
		ax1.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logPostOC"], color="#61D04F")
		ax1.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logPostOC"], color="#61D04F")
	
	else:
		ax1.bar(merged_sub["Rank"], merged_sub["logPostOC"], color="#61D04F")
		
	ax1.set(ylabel="logPostOC")
	ax1.margins(0.05, 0.2)
	ax1.set_xlim([0, args.top + 1])
	ax1.axhline(y=0,linewidth=2, color='k')


	if (args.highlight is not None) and (merged_sub["ID"].str.contains(args.highlight)):
		ax2.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logBF"], color="#2297E6")
		ax2.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logBF"], color="#2297E6")

	else:
		ax2.bar(merged_sub["Rank"], merged_sub["logBF"], color="#2297E6")

	ax2.set(ylabel="logBF")
	ax2.margins(0.05, 0.2)
	ax2.set_xlim([0, args.top + 1])
	ax2.axhline(y=0,linewidth=2, color='k')
	ax2.axhline(y=max_logBF, linewidth=2, color='k', linestyle='--')


	if args.highlight is not None and (merged_sub["ID"].str.contains(args.highlight)):
		ax3.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logPriorOC"], color="#DF536B")
		ax3.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logPriorOC"], color="#DF536B")

	else:
		ax3.bar(merged_sub["Rank"], merged_sub["logPriorOC"], color="#DF536B")

	ax3.set(xlabel="Rank", ylabel="logPriorOC")
	ax3.margins(0.05, 0.2)
	ax3.set_xlim([0, args.top + 1])
	ax3.axhline(y=0,linewidth=2, color='k')

	plt.savefig(args.outputDir + args.prefix + ".BICEP.png", dpi=300)








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

	logging.info("POSTERIOR")
	logging.info(" ")


	priorFile = None
	bfFile = None
	

	# command line arguments
	#if (args.prior is None) or (args.bf is None):
	priorFile = args.prefix + ".priors.txt"
	bfFile = args.prefix + ".BF.txt"

	#else:
	#	priorFile = args.prior
	#	bfFile = args.bf


	# load the prior and BF files
	logging.info("Reading in the prior file")
	with open(args.outputDir + priorFile) as f:
		prior = pd.read_csv(f, sep="\t", na_values=['.'])
		prior.dropna(subset=['prior'], inplace=True)
	
	
	logging.info("Reading in the Bayes factor file")
	with open(args.outputDir + bfFile) as f:
		bf = pd.read_csv(f, sep="\t", na_values=['.'])
		bf.dropna(subset=['BF'], inplace=True)
		bf = bf[bf["BF"] > 0]
	


	# combine values and output
	logging.info("Merge and output")
	merged = prior.merge(bf, on="ID")


	# if Prior == 0, set to min non-zero
	merged.loc[merged["prior"] == 0, "prior"] = np.min(merged[merged["prior"] > 0]["prior"])
	merged["PriorOC"] = merged["prior"] / (1 - merged["prior"] )
	merged["logPriorOC"] = np.log10(merged["PriorOC"])

	# drop NA values
	merged.replace([np.inf, -np.inf], np.nan, inplace=True)
	merged.dropna(subset=['prior', 'PriorOC', 'logPriorOC', 'BF', 'logBF'], inplace=True)


	# calculate the posteriors
	merged["logPriorOC"] = pd.to_numeric(merged["logPriorOC"], errors='coerce')
	merged["logPostOC"] = merged["logPriorOC"] + merged["logBF"]
	merged.dropna(subset=['logPostOC'], inplace=True)
	merged["Rank"] = merged["logPostOC"].rank(ascending=False, method='first')


	# get rid of unnecessary columns and reorder
	merged = merged.drop(['prior', 'PriorOC', 'BF'], axis=1)
	merged = merged.round(6)

	orderFirst = [ "Rank", "ID", "Gene", "csq", "logPostOC", "logPriorOC", "logBF", "STRING" ]
	orderSecond = [ x for x in merged.columns.tolist() and x not in order.first ]

	merged = merged[ orderFirst + orderSecond ]


	with open(args.tempDir + args.prefix + '.max_logBF.txt', 'r') as f:
		tmp = f.readlines()
		max_logBF = float(tmp[0])

	merged_sub = merged.sort_values(by=['logPostOC'], ascending=False).head(n=args.top)

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

	if (args.highlight is not None) and (merged_sub["ID"].str.contains(args.highlight).any()):
		ax1.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logPostOC"], color="#61D04F", hatch='//')
		ax1.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logPostOC"], color="#61D04F")
	
	else:
		ax1.bar(merged_sub["Rank"], merged_sub["logPostOC"], color="#61D04F")
		
	ax1.set(ylabel="logPostOC")
	ax1.margins(0.05, 0.2)
	ax1.set_xlim([0, args.top + 1])
	ax1.axhline(y=0,linewidth=2, color='k')


	if (args.highlight is not None) and (merged_sub["ID"].str.contains(args.highlight).any()):
		ax2.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logBF"], color="#2297E6", hatch='//')
		ax2.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logBF"], color="#2297E6")

	else:
		ax2.bar(merged_sub["Rank"], merged_sub["logBF"], color="#2297E6")

	ax2.set(ylabel="logBF")
	ax2.margins(0.05, 0.2)
	ax2.set_xlim([0, args.top + 1])
	ax2.axhline(y=0,linewidth=2, color='k')
	ax2.axhline(y=max_logBF, linewidth=2, color='k', linestyle='--')


	if args.highlight is not None and (merged_sub["ID"].str.contains(args.highlight).any()):
		ax3.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logPriorOC"], color="#DF536B", hatch='//')
		ax3.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logPriorOC"], color="#DF536B")

	else:
		ax3.bar(merged_sub["Rank"], merged_sub["logPriorOC"], color="#DF536B")

	ax3.set(xlabel="Rank", ylabel="logPriorOC")
	ax3.margins(0.05, 0.2)
	ax3.set_xlim([0, args.top + 1])
	ax3.axhline(y=0,linewidth=2, color='k')

	plt.savefig(args.outputDir + args.prefix + ".BICEP.png", dpi=300)




	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")




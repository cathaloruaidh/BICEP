import csv
import logging
import math
import os
import re
import sys

import numpy as np
import matplotlib as plt
import pandas as pd

import BayesFactor

# main function
def PO_main(args):

	priorFile = None
	bfFile = None
	

	# command line arguments
	if args.input is not None:
		priorFile = args.input + ".priors.txt"
		bfFile = args.input + ".BF.txt"

	if args.prior is not None:
		priorFile = args.prior

	if args.bf is not None:
		bfFile = args.bf


	# load the prior and BF files
	with open(args.outputDir + priorFile) as f:
		prior = pd.read_csv(f, sep="\t")
	
	with open(args.outputDir + bfFile) as f:
		bf = pd.read_csv(f, sep="\t")
	
	merged = prior.merge(bf, on="ID")
	merged["logPostOC"] = merged["logPriorOC"] + merged["logBF"]

	print(merged.sort_values(by=['logPostOC'], ascending = False).head())
	




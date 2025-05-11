#!/usr/bin/env python3


import argparse
import logging
import os
import os.path
import sys
import textwrap
import warnings


from argparse import SUPPRESS
from datetime import datetime
from threading import Lock


import Prior_Train
import Prior_Apply
import BayesFactor
import Posterior 







# create formatting class for argparse
class UltimateHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
	pass













# main function
def main(argv):
	

	warnings.filterwarnings("ignore")


	# argparse lobal arguments
	BICEP_textwrap = textwrap.dedent('''\
	----------------------------------------
	         _   ___   __   __   _ 
	        |_)   |   /    |_   |_)
	        |_)  _|_  \__  |__  |  

	        Bayesian Inference for 
	   Causality Evaluation in Pedigrees
	----------------------------------------
	 ''')

	parser = argparse.ArgumentParser(prog="BICEP",
	usage=SUPPRESS,
	formatter_class=UltimateHelpFormatter,
	description=BICEP_textwrap)
	parser._optionals.title = "Global arguments"

	sub_parsers = parser.add_subparsers(title="Sub-commands", dest='command')

	parser_parent = argparse.ArgumentParser(formatter_class=UltimateHelpFormatter, usage=SUPPRESS)
	parser_parent.add_argument("-l", "--log", nargs='?', default="INFO", help="Logging level: ERROR, WARN, INFO, DEBUG", choices=['ERROR', 'WARN', 'INFO', 'DEBUG'], metavar='STRING')
	parser_parent.add_argument("-n", "--cores", nargs='?', default=1, type=int, help="Number of CPU cores available", metavar='N')
	parser_parent.add_argument("--prefix", nargs='?', default="BICEP_output", help="Output prefix", metavar='STRING')
	parser_parent.add_argument("--build", nargs='?', default='GRCh38', help="Reference genome build: GRCh37, GRCh38", choices=['GRCh37', 'GRCh38'], metavar='STRING')
	parser_parent.add_argument("--frequency", nargs='?', default="gnomADe_AF", help="Allele frequency predictor", metavar='STRING')
	parser_parent.add_argument("--cnv", action='store_true', help="Use the CNV prior models")
	parser_parent.add_argument("-k", "--key", nargs='?', help="VCF FORMAT tag for genotype or carrier status", metavar='STRING')


	# All sub-command
	parser_ALL = sub_parsers.add_parser("All", help = "Run all steps of BICEP", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\

	Run all steps of BICEP'''))
	parser_ALL.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='FILE')
	parser_ALL.add_argument("--clinvar", nargs='?', help="ClinVar VCF file annotated with VEP", metavar='FILE')
	parser_ALL.add_argument("--clinvarPrefix", nargs='?', help="Prefix for ClinVar VCF file in data directory", metavar='FILE')
	parser_ALL.add_argument("-e", "--exclude", nargs='?', help="File of ClinVar IDs to exclude from training", metavar='FILE')
	parser_ALL.add_argument("-i", "--include", nargs='?', help="File of ClinVar IDs to include for training", metavar='FILE')
	parser_ALL.add_argument("-b", "--benign", nargs='?', help="File of benign variant IDs for training", metavar='FILE')
	parser_ALL.add_argument("-p", "--pathogenic", nargs='?', help="File of pathogenic variant IDs for training", metavar='FILE')
	parser_ALL.add_argument("--eval", action='store_true', help="Evaluate the predictors and regression model for the prior")
	parser_ALL.add_argument("--boot", nargs='?', default=1000, type=int, help="Number of bootstraps for prior evaluation", metavar='N')
	parser_ALL.add_argument("-m", "--model", nargs='?', help="Prefix for the regression model files", metavar='STRING')
	parser_ALL.add_argument("-v", "--vcf", nargs='?', help="VCF file for variants", metavar='FILE', required = True)
	parser_ALL.add_argument("-f", "--fam", nargs='?', help="FAM file describing the pedigree structure and phenotypes", metavar='FILE', required = True)
	parser_ALL.add_argument("--priorCaus", nargs='?', default="linear", choices=["uniform", "linear"], help="Prior parameter distribution for causal model", metavar='STRING')
	parser_ALL.add_argument("--priorNeut", nargs='?', default="uniform", choices=["uniform", "linear"], help="Prior parameter distribution for neutral model", metavar='STRING')
	parser_ALL.add_argument("--top", nargs='?', default=50, type=int, help="Number of top ranking variants to plot", metavar='N')
	parser_ALL.add_argument("--highlight", nargs='?', help="ID of variant to highlight in plot", metavar='STRING')
	
	


	# PriorTrain sub-command
	parser_PT = sub_parsers.add_parser("PriorTrain", help = "Train the regresison models to generate a prior", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\

	Train the regresison models to generate a prior'''))
	parser_PT.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='FILE')
	parser_PT.add_argument("--clinvar", nargs='?', help="ClinVar VCF file annotated with VEP", metavar='FILE')
	parser_PT.add_argument("--clinvarPrefix", nargs='?', help="Prefix for ClinVar VCF file in data directory", metavar='FILE')
	parser_PT.add_argument("-v", "--vcf", nargs='?', help="VCF file for variants", metavar='FILE', required = True)
	parser_PT.add_argument("-e", "--exclude", nargs='?', help="File of ClinVar IDs to exclude from training", metavar='FILE')
	parser_PT.add_argument("-i", "--include", nargs='?', help="File of ClinVar IDs to include for training", metavar='FILE')
	parser_PT.add_argument("-b", "--benign", nargs='?', help="File of benign variant IDs for training", metavar='FILE')
	parser_PT.add_argument("-p", "--pathogenic", nargs='?', help="File of pathogenic variant IDs for training", metavar='FILE')
	parser_PT.add_argument("--eval", action='store_true', help="Evaluate the predictors and regression model for the prior")
	parser_PT.add_argument("--boot", nargs='?', default=1000, type=int, help="Number of bootstraps for prior evaluation", metavar='N')
	


	
	
	# PriorApply sub-command
	parser_PA = sub_parsers.add_parser("PriorApply", help = "Apply the regression models to the pedigree data", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\
	
	Apply the regression models to the pedigree data'''))
	parser_PA.add_argument("-v", "--vcf", nargs='?', help="VCF file for variants", metavar='FILE', required = True)
	parser_PA.add_argument("-m", "--model", nargs='?', help="Prefix for the regression model files", metavar='STRING')
	parser_PA.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='FILE')
	
	
	
	# BF sub-command
	parser_BF = sub_parsers.add_parser("BayesFactor", help = "Calculate Bayes factors for co-segregation", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description= BICEP_textwrap + textwrap.dedent('''\

	Calculate Bayes factors for co-segregation'''))
	parser_BF.add_argument("-f", "--fam", nargs='?', help="FAM file describing the pedigree structure and phenotypes", metavar='FILE', required = True)
	parser_BF.add_argument("-v", "--vcf", nargs='?', help="VCF file for variants", metavar='FILE', required = True)
	parser_BF.add_argument("--priorCaus", nargs='?', default="linear", choices=["uniform", "linear"], help="Prior parameter distribution for causal model", metavar='STRING')
	parser_BF.add_argument("--priorNeut", nargs='?', default="uniform", choices=["uniform", "linear"], help="Prior parameter distribution for neutral model", metavar='STRING')



	# Posterior sub-command
	parser_PO = sub_parsers.add_parser("Posterior", help = "Generate posteriors and plots", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\
	
	Generate posteriors and plots'''))
	parser_PO.add_argument("--prior", nargs='?', help="Prefix for the prior input", metavar='FILE')
	parser_PO.add_argument("--bf", nargs='?', help="Prefix for the Bayes factor input", metavar='FILE')
	parser_PO.add_argument("--top", nargs='?', default=50, type=int, help="Number of top ranking variants to plot", metavar='N')
	parser_PO.add_argument("--highlight", nargs='?', help="ID of variant to highlight in plot", metavar='STRING')
	parser_PO.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='FILE')
	

	args = parser.parse_args()


	if args.command is not None:

		# directory paths
		args.outputDir = "BICEP_results/"
		args.tempDir = "BICEP_temp/"
		args.logDir = "BICEP_log/"
		args.scriptDir = os.path.dirname(os.path.realpath(__file__)) + "/"


		# create directories
		if not os.path.exists(args.outputDir):
			os.makedirs(args.outputDir)

		if not os.path.exists(args.tempDir):
			os.makedirs(args.tempDir)

		if not os.path.exists(args.logDir):
			os.makedirs(args.logDir)



		# set logging details
		now = datetime.now()

		logLevel = args.log.upper()
		FORMAT = '# %(asctime)s [%(levelname)s] - %(message)s'
		logging.basicConfig(level=logLevel, format=FORMAT)

		# add colours to log name
		logging.addLevelName(logging.NOTSET, "NOT  ")
		logging.addLevelName(logging.DEBUG, "\u001b[36mDEBUG\u001b[0m")
		logging.addLevelName(logging.INFO, "INFO ")
		logging.addLevelName(logging.WARNING, "\u001b[33mWARN \u001b[0m")
		logging.addLevelName(logging.ERROR, "\u001b[31mERROR\u001b[0m")
		logging.addLevelName(logging.CRITICAL, "\u001b[35mCRIT \u001b[0m")

		rootLogger = logging.getLogger()
		logFormatter = logging.Formatter(FORMAT)

		fileHandler = logging.FileHandler("{0}/{1}.{2}.{3}.log".format(args.logDir, args.prefix, args.command, now.strftime("%Y_%m_%d-%H_%M_%S")))
		fileHandler.setFormatter(logFormatter)
		rootLogger.addHandler(fileHandler)

		#consoleHandler = logging.StreamHandler()
		#consoleHandler.setFormatter(logFormatter)
		#rootLogger.addHandler(consoleHandler)

		#if args.model is None:
		#	args.model = args.tempDir + args.prefix

		if args.command == "All":

			if args.model is None:
				args.model = args.tempDir + args.prefix
				Prior_Train.PT_main(args)

			if args.clinvar is None:
				args.clinvar = args.scriptDir + "../data/clinvar_20231126." + args.build +".PATH_BEN.single.strip.vep.vcf.gz"

			Prior_Apply.PA_main(args)
			BayesFactor.BF_main(args)
			Posterior.PO_main(args)
			

		if args.command == "PriorTrain":
			Prior_Train.PT_main(args)

		if args.command == "PriorApply":
			Prior_Apply.PA_main(args)

		if args.command == "BayesFactor":
			BayesFactor.BF_main(args)

		if args.command == "Posterior":
			Posterior.PO_main(args)


if __name__ == "__main__":
	main(sys.argv[1:])

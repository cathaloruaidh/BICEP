#!/usr/bin/python3


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
	parser_parent.add_argument("-l", "--log", nargs='?', default="INFO", help="Logging level: ERROR, WARN, INFO, DEBUG", choices=['ERROR', 'WARN', 'INFO', 'DEBUG'], metavar='C')
	parser_parent.add_argument("-n", "--cores", nargs='?', default=1, type=int, help="Number of CPU cores available", metavar='N')
	parser_parent.add_argument("--prefix", nargs='?', default="BICEP_output", help="Output prefix", metavar='C')



	# PriorTrain sub-command
	parser_PT = sub_parsers.add_parser("PriorTrain", help = "Train the regresison models to generate a prior", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\

	Train the regresison models to generate a prior'''))
	parser_PT.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='C')
	parser_PT.add_argument("--clinvar", nargs='?', help="ClinVar VCF file annotated with VEP", metavar='C')
	parser_PT.add_argument("-e", "--exclude", nargs='?', help="File of ClinVar IDs to exclude from training", metavar='C')
	parser_PT.add_argument("-i", "--include", nargs='?', help="File of ClinVar IDs to include for training", metavar='C')
	parser_PT.add_argument("-b", "--benign", nargs='?', help="File of benign variant IDs for training", metavar='C')
	parser_PT.add_argument("-p", "--pathogenic", nargs='?', help="File of pathogenic variant IDs for training", metavar='C')

	
	
	# PriorEvaluate sub-command
	parser_PE = sub_parsers.add_parser("PriorEvaluate", help = "Evaluate the performance of the regression models", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\
	
	Evaluate the performance of the regression models'''))
	parser_PE.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='C')
	parser_PE.add_argument("--boot", nargs='?', default=1, type=int, help="Number of bootstraps", metavar='N')
	parser_PE.add_argument("--clinvar", nargs='?', help="ClinVar VCF file annotated with VEP", metavar='C')
	


	# PriorApply sub-command
	parser_PA = sub_parsers.add_parser("PriorApply", help = "Apply the regression models to the pedigree data", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\
	
	Apply the regression models to the pedigree data'''))
	parser_PA.add_argument("-v", "--vcf", nargs='?', help="VCF file for variants", metavar='F', required = True)
	parser_PA.add_argument("-m", "--model", nargs='?', help="Prefix for the regression model files", metavar='C')
	parser_PA.add_argument("--predictors", nargs='?', help="File containing regression predictors", metavar='C')
	
	
	
	# BF sub-command
	parser_BF = sub_parsers.add_parser("BayesFactor", help = "Calculate Bayes factors for co-segregation", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description= BICEP_textwrap + textwrap.dedent('''\

	Calculate Bayes factors for co-segregation'''))
	parser_BF.add_argument("-f", "--fam", nargs='?', help="FAM file describing the pedigree structure and phenotypes", metavar='F', required = True)
	parser_BF.add_argument("-v", "--vcf", nargs='?', help="VCF file for variants", metavar='F', required = True)
	parser_BF.add_argument("--minAff", nargs='?', default=0, type=int, help="Minimum affected individuals per pedigree", metavar='N')
	parser_BF.add_argument("--priorCaus", nargs='?', default="linear", choices=["uniform", "linear"], help="Prior parameter distribution for causal model", metavar='C')
	parser_BF.add_argument("--priorNeut", nargs='?', default="uniform", choices=["uniform", "linear"], help="Prior parameter distribution for neutral model", metavar='C')



	# Posterior sub-command
	parser_Post = sub_parsers.add_parser("Posterior", help = "Generate posteriors and plots", 
	parents = [parser_parent], add_help=False, formatter_class=UltimateHelpFormatter, usage=SUPPRESS, 
	description = BICEP_textwrap + textwrap.dedent('''\
	
	Generate posteriors and plots'''))
	parser_PO.add_argument("--input", nargs='?', help="Common prefix for the Bayes factor and prior input", metavar='C')
	parser_PO.add_argument("--prior", nargs='?', help="Prefix for the prior input", metavar='C')
	parser_PO.add_argument("--bf", nargs='?', help="Prefix for the Bayes factor input", metavar='C')
	

	args = parser.parse_args()


	if args.command is not None:

		# directory paths
		args.outputDir = "BICEP_results/"
		args.tempDir = "BICEP_temp/"
		args.logDir = "BICEP_log/"


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

		fileHandler = logging.FileHandler("{0}/{1}.{2}.{3}.log".format(args.logDir, args.output, args.command, now.strftime("%Y_%m_%d-%H_%M_%S")))
		fileHandler.setFormatter(logFormatter)
		rootLogger.addHandler(fileHandler)

		#consoleHandler = logging.StreamHandler()
		#consoleHandler.setFormatter(logFormatter)
		#rootLogger.addHandler(consoleHandler)


		if args.command == "PriorTrain":
			Prior_Train.PT_main(args)

		# not working currently!
		if args.command == "PriorEvaluate":
			Prior_Evaluate.PE_main(args)

		if args.command == "PriorApply":
			Prior_Apply.PA_main(args)

		if args.command == "BayesFactor":
			BayesFactor.BF_main(args)

		if args.command == "Posterior":
			Posterior.PO_main(args)


if __name__ == "__main__":
	main(sys.argv[1:])

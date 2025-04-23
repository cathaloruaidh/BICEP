import glob
import logging
import math
import os
import os.path
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cyvcf2 import VCF, Writer
from itertools import zip_longest
from joblib import dump, load
from multiprocessing import cpu_count
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, r2_score
from sklearn.preprocessing import MinMaxScaler







# main function
def PA_main(args):


	logging.info("PRIOR - APPLY")
	logging.info(" ")


	# ignore warnings
	warnings.filterwarnings("ignore")
	pd.set_option('display.max_columns', None)



	# command line arguments

	nCores = 1
	inputVcfFile = None
	logLevel = None
	modelPrefix = None
	outputPrefix = None
	predictorsFile = None


	if args.cores is not None:
		if int(args.cores) <= cpu_count():
			nCores = int(args.cores)

	if args.model is not None:
		modelPrefix = args.model

	if args.prefix is not None:
		outputPrefix = args.prefix

	if args.vcf is not None:
		inputVcfFile = args.vcf

	if args.predictors is not None:
		predictorsFile = args.predictors
	

	# check if model files exist
	if modelPrefix is None:
		logging.error("No model files specified")
		sys.exit(1)


	if not glob.glob(modelPrefix + "*.pkl"):
		logging.critical("No model files found")
		sys.exit(1)


	# define VEP hierarchy for ClinVar consequences examined
	vepCSQRank = {'transcript_ablation' : 1,
	'splice_acceptor_variant' : 2,
	'splice_donor_variant' : 3,
	'stop_gained' : 4,
	'frameshift_variant' : 5,
	'stop_lost' : 6,
	'start_lost' : 7,
	'transcript_amplification' : 8,
	'inframe_insertion' : 9,
	'inframe_deletion' : 10,
	'missense_variant' : 11,
	'protein_altering_variant' : 12,
	'splice_region_variant' : 13,
	'splice_donor_5th_base_variant' : 14,
	'splice_donor_region_variant' : 15,
	'splice_polypyrimidine_tract_variant' : 16,
	'incomplete_terminal_codon_variant' : 17,
	'start_retained_variant' : 18,
	'stop_retained_variant' : 19,
	'synonymous_variant' : 20,
	'coding_sequence_variant' : 21,
	'mature_miRNA_variant' : 22,
	'5_prime_UTR_variant' : 23,
	'3_prime_UTR_variant' : 24,
	'non_coding_transcript_exon_variant' : 25,
	'intron_variant' : 26,
	'NMD_transcript_variant' : 27,
	'non_coding_transcript_variant' : 28,
	'upstream_gene_variant' : 29,
	'downstream_gene_variant' : 30,
	'TFBS_ablation' : 31,
	'TFBS_amplification' : 32,
	'TF_binding_site_variant' : 33,
	'regulatory_region_ablation' : 34,
	'regulatory_region_amplification' : 35,
	'feature_elongation' : 36,
	'regulatory_region_variant' : 37,
	'feature_truncation' : 38,
	'intergenic_variant' : 39}


	vepCSQRankCoding = {'transcript_ablation' : 1,
	'splice_acceptor_variant' : 2,
	'splice_donor_variant' : 3,
	'stop_gained' : 4,
	'frameshift_variant' : 5,
	'stop_lost' : 6,
	'start_lost' : 7,
	'transcript_amplification' : 8,
	'inframe_insertion' : 9,
	'inframe_deletion' : 10,
	'missense_variant' : 11,
	'protein_altering_variant' : 12,
	'splice_region_variant' : 13,
	'start_retained_variant' : 14,
	'stop_retained_variant' : 15,
	'synonymous_variant' : 16,
	'coding_sequence_variant' : 17,
	'5_prime_UTR_variant' : 18,
	'3_prime_UTR_variant' : 19}


	vepIMPACTRank = { 'HIGH' : 1,
	'MODERATE' : 2,
	'LOW' : 3,
	'MODIFIER' : 4}


	vepCSQImpact = {'transcript_ablation' : 'HIGH',
	'splice_acceptor_variant' : 'HIGH',
	'splice_donor_variant' : 'HIGH',
	'stop_gained' : 'HIGH',
	'frameshift_variant' : 'HIGH',
	'stop_lost' : 'HIGH',
	'start_lost' : 'HIGH',
	'transcript_amplification' : 'HIGH',
	'inframe_insertion' : 'MODERATE',
	'inframe_deletion' : 'MODERATE',
	'missense_variant' : 'MODERATE',
	'protein_altering_variant' : 'MODERATE',
	'splice_region_variant' : 'LOW',
	'splice_donor_5th_base_variant' : 'LOW',
	'splice_donor_region_variant' : 'LOW',
	'splice_polypyrimidine_tract_variant' : 'LOW',
	'incomplete_terminal_codon_variant' : 'LOW',
	'start_retained_variant' : 'LOW',
	'stop_retained_variant' : 'LOW',
	'synonymous_variant' : 'LOW',
	'coding_sequence_variant' : 'MODIFIER',
	'mature_miRNA_variant' : 'MODIFIER',
	'5_prime_UTR_variant' : 'MODIFIER',
	'3_prime_UTR_variant' : 'MODIFIER',
	'non_coding_transcript_exon_variant' : 'MODIFIER',
	'intron_variant' : 'MODIFIER',
	'NMD_transcript_variant' : 'MODIFIER',
	'non_coding_transcript_variant' : 'MODIFIER',
	'upstream_gene_variant' : 'MODIFIER',
	'downstream_gene_variant' : 'MODIFIER',
	'TFBS_ablation' : 'MODIFIER',
	'TFBS_amplification' : 'MODIFIER',
	'TF_binding_site_variant' : 'MODIFIER',
	'regulatory_region_ablation' : 'MODIFIER',
	'regulatory_region_amplification' : 'MODIFIER',
	'feature_elongation' : 'MODIFIER',
	'regulatory_region_variant' : 'MODIFIER',
	'feature_truncation' : 'MODIFIER',
	'intergenic_variant' : 'MODIFIER'}




	# get flat priors from training data	
	flatPriors = {}
	if os.path.isfile(modelPrefix + '.flatPriors.pkl'):
		with open(modelPrefix + '.flatPriors.pkl', 'rb') as f:
			flatPriors = pickle.load(f)





	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")



	vcf = VCF(inputVcfFile, gts012=True)


	keys = re.sub('^.*?: ', '', vcf.get_header_type('CSQ')['Description']).split("|")
	keys = [ key.strip("\"") for key in keys ]


	if predictorsFile is not None:

		d = {}



		# manually add allele frequency to the predictors
		if args.cnv:
			with open(args.predictors, 'r') as f:
				for line in f:
					(model, key)=line.split()
					d[model, key] = 1

			# allele frequency predictor for prior
			if args.frequency is None:
				args.frequency = "gnomAD_v4.1_Max_PopMax_AF"

			# manually add allele frequency to the predictors
			keysPredictors = [ x[1] for x in d.keys()] + [ args.frequency ]
			keysPredictors = sorted(list(set(keysPredictors)))

			keysPredictors_DEL = [ f for m,f in d.keys() if m == "DEL" ] + [args.frequency]
			keysPredictors_DUP = [ f for m,f in d.keys() if m == "DUP" ] + [args.frequency]


		else:
			with open(args.predictors, 'r') as f:
				for line in f:
					(model, key, value)=line.split()
					d[model, key] = value

			# allele frequency predictor for prior
			if args.frequency is None:
				args.frequency = "gnomAD_v2_exome_AF_popmax"

			keysPredictors = sorted([ x[1] for x in d.keys()])
			keysDescPred = sorted([ f for m,f in d.keys() if d[m,f] == "L" ] + [args.frequency])
			keysAscPred = sorted([ f for m,f in d.keys() if d[m,f] == "H" ])
			keysPredictors = sorted( keysDescPred + keysAscPred )

			keysPredictors_IND = [ f for m,f in d.keys() if m == "IND" ] + [args.frequency]
			keysPredictors_MIS = [ f for m,f in d.keys() if m == "MIS" ] + [args.frequency]
			keysPredictors_OTH = [ f for m,f in d.keys() if m == "OTH" ] + [args.frequency]




	else:
		if args.cnv:
			keysPredictors = sorted([ "overlap_loeuf_sumRecip",  args.frequency ])
			keysPredictors_DEL = keysPredictors
			keysPredictors_DUP = keysPredictors

		else:
			keysPredictors = sorted([ "CADD_PHRED", "FATHMM_score", "MPC_score", "Polyphen2_HDIV_score", "REVEL_score", "SIFT_score" ] + [args.frequency])
			keysDescPred = sorted([ "SIFT_score", "FATHMM_score" ] + [args.frequency])
			keysAscPred  = sorted([ x for x in keysPredictors if x not in keysDescPred ])

			keysPredictors_IND = sorted([args.frequency])
			keysPredictors_MIS = sorted(["FATHMM_score", "MPC_score", "Polyphen2_HDIV_score", "REVEL_score", "SIFT_score"] + [args.frequency])
			keysPredictors_OTH = [args.frequency]
	



	logging.info("Converting annotation data to list")

	# save variants in list 
	DATA = []
	flat_DATA = []
	CHROMS = set([ "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22" ])

	for variant in vcf:
		# remove variants not on autosomes
		if len( set([variant.CHROM, "chr"+variant.CHROM]) & CHROMS ) == 0:
			continue

		if args.cnv:
			# get the ID of the variant
			ID = variant.ID
			#ID = variant.CHROM + "_" + str(variant.start+1) + "_" + str(variant.INFO.get('END')) + "_" + variant.INFO.get('SVTYPE')

			msg = "Reading variant " + ID
			logging.debug(msg)


			SVLEN = variant.INFO.get('SVLEN')


			# get the variant type, ignore if not SNV or indel
			typeTMP = variant.INFO.get('SVTYPE')
			typeVEP = "OTHER"

			if (typeTMP == "DEL" or typeTMP == "DUP") and abs(SVLEN) >= 50:
				typeVEP = typeTMP


			# get vep consequences and create a dictionary
			CSQ = np.array(variant.INFO.get('CSQ').split("|"))
			if CSQ is not None:

				csqVEP = dict(zip(keys, CSQ.T))	
				l = [ ID, typeVEP, SVLEN ]

				for key in keysPredictors:
					l.append(csqVEP[key])

				DATA.append(l) 


			# otherwise flat, nonCoding prior
			else:
				if 'nonCoding' in flatPriors.keys():
					flat_DATA.append([ID, flatPriors['nonCoding']])

				else:
					flat_DATA.append([ID, np.nan])

		else:
			# get the ID of the variant
			ID = variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0] 

			msg = "Reading variant " + ID
			logging.debug(msg)



			# get the variant type, ignore if not SNV or indel
			if len(variant.REF) == 1 and len(variant.ALT[0]) == 1:
				typeVEP = "SNV"
			elif ( (len(variant.REF) == 1 and len(variant.ALT[0]) > 1) or ((len(variant.REF) > 1 and len(variant.ALT[0]) == 1)) ) and max(len(variant.REF), len(variant.ALT[0])) < 50:
				typeVEP = "indel"



			# get vep consequences and create a dictionary
			CSQ = variant.INFO.get('CSQ').split(",")


			csqVEP = []


			for i in range(len(CSQ)):
				add = True
				tmp = np.array(CSQ[i].split("|"))
				dictVEP = dict(zip_longest(keys, tmp.T, fillvalue="."))


				# split VEP consequences if multiple
				if "&" in dictVEP["Consequence"]:
					dictVEP["Consequence_split"] = dictVEP["Consequence"].split("&")
				
				else:
					dictVEP["Consequence_split"] = [ dictVEP["Consequence"] ]


				# select the consequence with the highest vep rank
				csqCVSubset = [ x for x in dictVEP["Consequence_split"] if x in vepCSQRankCoding.keys() ]
				
				if len(csqCVSubset) > 0:
					d = dict((k, vepCSQRank[k]) for k in csqCVSubset)
					dictVEP["Consequence_select"] = min(d, key=d.get)
				else:
					add = False


				# extract the transcript-specific metrics from dbNSFP
				# and pick the most deleterious value
				for key in keysAscPred:
					l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
					if len(l) == 0:
						dictVEP[key] = np.nan
					else:
						dictVEP[key] = max(l)

				for key in keysDescPred:
					l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
					if len(l) == 0:
						dictVEP[key] = np.nan
					else:
						dictVEP[key] = min(l)

				

				# if the transcript is not protein-coding reject it
				if dictVEP['BIOTYPE'] != "protein_coding":
					add = False


				# make sure transcript is canonical
				if dictVEP['CANONICAL'] != "YES":
					add = False

				
				# if the transcript hasn't been rejected, add it to the list
				if add:
					csqVEP.append(dictVEP)



			# summarize scores across all transcripts
			if len(csqVEP) > 0:
				for i in range(len(csqVEP)):
					l = [ ID, csqVEP[i]["SYMBOL"], typeVEP, csqVEP[i]["Consequence_select"], csqVEP[i]["IMPACT"] ]

					for key in keysAscPred + keysDescPred:
						l.append(csqVEP[i][key])

					DATA.append(l) 


			# otherwise flat, nonCoding prior
			else:
				if 'nonCoding' in flatPriors.keys():
					flat_DATA.append([ID, flatPriors['nonCoding']])

				else:
					flat_DATA.append([ID, np.nan])


	start = 3 if args.cnv else 5

	for i in range(len(DATA)):
		for j in range(start, len(DATA[i])):
			if DATA[i][j] == "." or DATA[i][j] == "":
				DATA[i][j] = np.nan
			else:
				DATA[i][j] = float(DATA[i][j])

	logging.info(" ")




	################################################################################
	# Apply the regression prediction
	################################################################################

	pd.set_option('display.max_colwidth', 100)

	# read in the data
	if args.cnv:
		df = pd.DataFrame(DATA, columns = ['ID', 'typeVEP', 'SVLEN'] + keysPredictors )
		df[args.frequency] = df[args.frequency].fillna(0.0)

		x = df.filter( ['typeVEP'] + keysPredictors )



		## DEL
		logging.info("DEL")

		x_DEL = x[ x['typeVEP'] == 'DEL' ]
		x_DEL_index = x_DEL.index

		y_DEL = df[ df['typeVEP'] == 'DEL' ]



		if os.path.isfile(modelPrefix + '.DEL_predictors.npy') and not x_DEL.empty:

			# load in the predictor IDs for the models
			with open(modelPrefix + '.DEL_predictors.npy', 'rb') as f:
				predictors_DEL = np.load(f, allow_pickle = True)
				predictors_DEL = [ s.replace('CV', 'VEP') for s in predictors_DEL ]

			x_DEL = x_DEL[predictors_DEL]



			# impute the missing data
			logging.info("Impute the data")

			with open(modelPrefix + '.DEL_imp.pkl', 'rb') as f:
				imp_DEL = pickle.load(f)

			x_DEL_imp = imp_DEL.transform(x_DEL)



			# scale the data
			logging.info("Scaling to [0,1]")

			with open(modelPrefix + '.DEL_scal.pkl', 'rb') as f:
				scal_DEL = pickle.load(f)

			x_DEL_imp_scal = scal_DEL.transform(x_DEL_imp)


			# run logistic regression
			logging.info("Apply logistic regression")

			with open(modelPrefix + '.DEL_logReg.pkl', 'rb') as f:
				logReg_DEL = pickle.load(f)


			y_DEL_pred = logReg_DEL.predict_proba(x_DEL_imp_scal)[:,1]


		else:
			if 'DEL' in flatPriors.keys():
				logging.info("Using flat priors for DEL. ")
				y_DEL_pred = np.full(len(x_DEL), flatPriors['DEL'])
			else:
				logging.info("Ignoring all DEL. ")
				y_DEL_pred = np.full(len(x_DEL), np.nan)




		## DUP
		logging.info("DUP")

		x_DUP = x[ x['typeVEP'] == 'DUP' ]
		x_DUP_index = x_DUP.index

		y_DUP = df[ df['typeVEP'] == 'DUP' ]



		if os.path.isfile(modelPrefix + '.DUP_predictors.npy') and not x_DUP.empty:

			# load in the predictor IDs for the models
			with open(modelPrefix + '.DUP_predictors.npy', 'rb') as f:
				predictors_DUP = np.load(f, allow_pickle = True)
				predictors_DUP = [ s.replace('CV', 'VEP') for s in predictors_DUP ]

			x_DUP = x_DUP[predictors_DUP]



			# impute the missing data
			logging.info("Impute the data")

			with open(modelPrefix + '.DUP_imp.pkl', 'rb') as f:
				imp_DUP = pickle.load(f)

			x_DUP_imp = imp_DUP.transform(x_DUP)



			# scale the data
			logging.info("Scaling to [0,1]")

			with open(modelPrefix + '.DUP_scal.pkl', 'rb') as f:
				scal_DUP = pickle.load(f)

			x_DUP_imp_scal = scal_DUP.transform(x_DUP_imp)


			# run logistic regression
			logging.info("Apply logistic regression")

			with open(modelPrefix + '.DUP_logReg.pkl', 'rb') as f:
				logReg_DUP = pickle.load(f)


			y_DUP_pred = logReg_DUP.predict_proba(x_DUP_imp_scal)[:,1]


		else:
			if 'DUP' in flatPriors.keys():
				logging.info("Using flat priors for DUP. ")
				y_DUP_pred = np.full(len(x_DUP), flatPriors['DUP'])
			else:
				logging.info("Ignoring all DUP. ")
				y_DUP_pred = np.full(len(x_DUP), np.nan)


		logging.info(" ")



	else:
		df = pd.DataFrame(DATA, columns = ['ID', 'Gene', 'typeVEP', 'csqVEP', 'impactVEP'] + keysAscPred + keysDescPred )
		df[args.frequency] = df[args.frequency].fillna(0.0)

		df['csqVEP'] = pd.Categorical(df['csqVEP'], categories = sorted(vepCSQRank.keys()))
		df['impactVEP'] = pd.Categorical(df['impactVEP'], categories = sorted(vepIMPACTRank.keys()))
		df['typeVEP'] = pd.Categorical(df['typeVEP'], categories = ['indel', 'SNV'])


		x = df.filter( ['csqVEP', 'impactVEP', 'typeVEP'] + keysAscPred + keysDescPred)
		


		## INDELS
		logging.info("IND")

		x_IND = x[ x['typeVEP'] == 'indel' ]
		x_IND_csq = x_IND['csqVEP']
		x_IND_impact = x_IND['impactVEP']
		x_IND = pd.get_dummies(x_IND, columns = ['impactVEP'])
		x_IND_index = x_IND.index

		y_IND = df[ df['typeVEP'] == 'indel' ]



		if os.path.isfile(modelPrefix + '.IND_predictors.npy') and not x_IND.empty:

			# load in the predictor IDs for the models
			with open(modelPrefix + '.IND_predictors.npy', 'rb') as f:
				predictors_IND = np.load(f, allow_pickle = True)
				predictors_IND = [ s.replace('CV', 'VEP') for s in predictors_IND ]

			x_IND = x_IND[predictors_IND]



			# impute the missing data
			logging.info("Impute the data")

			with open(modelPrefix + '.IND_imp.pkl', 'rb') as f:
				imp_IND = pickle.load(f)

			x_IND_imp = imp_IND.transform(x_IND)



			# scale the data
			logging.info("Scaling to [0,1]")

			with open(modelPrefix + '.IND_scal.pkl', 'rb') as f:
				scal_IND = pickle.load(f)

			x_IND_imp_scal = scal_IND.transform(x_IND_imp)


			# run logistic regression
			logging.info("Apply logistic regression")

			with open(modelPrefix + '.IND_logReg.pkl', 'rb') as f:
				logReg_IND = pickle.load(f)


			y_IND_pred = logReg_IND.predict_proba(x_IND_imp_scal)[:,1]


		else:
			if 'IND' in flatPriors.keys():
				logging.info("Using flat priors for IND. ")
				y_IND_pred = np.full(len(x_IND), flatPriors['IND'])
			else:
				logging.info("Ignoring all IND. ")
				y_IND_pred = np.full(len(x_IND), np.nan)


		logging.info(" ")



		## MISSENSE SNV
		logging.info("MIS")

		x_MIS = x[ x['csqVEP'] == 'missense_variant' ]
		x_MIS_csq = x_MIS['csqVEP']
		x_MIS_impact = x_MIS['impactVEP']
		x_MIS_index = x_MIS.index


		y_MIS = df[ df['csqVEP'] == 'missense_variant' ]

		if os.path.isfile(modelPrefix + '.MIS_predictors.npy'):

			# load in the predictor IDs for the models
			with open(modelPrefix + '.MIS_predictors.npy', 'rb') as f:
				predictors_MIS = np.load(f, allow_pickle = True)
				predictors_MIS = [ s.replace('CV', 'VEP') for s in predictors_MIS ]

			x_MIS = x_MIS[predictors_MIS]


			# impute the missing data
			logging.info("Impute the data")

			with open(modelPrefix + '.MIS_imp.pkl', 'rb') as f:
				imp_MIS = pickle.load(f)


			x_MIS_imp = imp_MIS.transform(x_MIS)
			x_MIS_imp_df = pd.DataFrame(x_MIS_imp, columns = x_MIS.columns)


			# scale the data
			logging.info("Scaling to [0,1]")

			with open(modelPrefix + '.MIS_scal.pkl', 'rb') as f:
				scal_MIS = pickle.load(f)

			x_MIS_imp_scal = scal_MIS.transform(x_MIS_imp)

			x_MIS_imp_scal_df = pd.DataFrame(x_MIS_imp_scal, columns = x_MIS.columns)

			# run logistic regression
			logging.info("Apply logistic regression")

			with open(modelPrefix + '.MIS_logReg.pkl', 'rb') as f:
				logReg_MIS = pickle.load(f)

			y_MIS_pred = logReg_MIS.predict_proba(x_MIS_imp_scal)[:,1]

		
		else:
			if 'MIS' in flatPriors.keys():
				logging.info("Using flat priors for MIS. ")
				y_MIS_pred = np.full(len(x_MIS), flatPriors['MIS'])
			else:
				logging.info("Ignoring all MIS")
				y_MIS_pred = np.full(len(x_MIS), np.nan)

		logging.info(" ")



		## NON-MISSENSE SNV
		logging.info("NON-MISSENSE SNV")

		x_OTH = x[ (x['csqVEP'] != 'missense_variant') & (x['typeVEP'] == 'SNV') ]
		x_OTH_csq = x_OTH['csqVEP']
		x_OTH_impact = x_OTH['impactVEP']
		x_OTH = pd.get_dummies(x_OTH, columns = ['csqVEP'])
		x_OTH_index = x_OTH.index

		y_OTH = df[ (df['csqVEP'] != 'missense_variant') & (df['typeVEP'] == 'SNV') ]

			
		if os.path.isfile(modelPrefix + '.OTH_predictors.npy'):

			# load in the predictor IDs for the models
			with open(modelPrefix + '.OTH_predictors.npy', 'rb') as f:
				predictors_OTH = np.load(f, allow_pickle = True)
				predictors_OTH = [ s.replace('CV', 'VEP') for s in predictors_OTH ]

			x_OTH = x_OTH[predictors_OTH]


			# impute the missing data
			logging.info("Impute the data")

			with open(modelPrefix + '.OTH_imp.pkl', 'rb') as f:
				imp_OTH = pickle.load(f)
			
			x_OTH_imp = imp_OTH.transform(x_OTH)



			# scale the data
			logging.info("Scaling to [0,1]")

			with open(modelPrefix + '.OTH_scal.pkl', 'rb') as f:
				scal_OTH = pickle.load(f)

			x_OTH_imp_scal = scal_OTH.transform(x_OTH_imp)



			# run logistic regression
			logging.info("Apply logistic regression")

			with open(modelPrefix + '.OTH_logReg.pkl', 'rb') as f:
				logReg_OTH = pickle.load(f)


			y_OTH_pred = logReg_OTH.predict_proba(x_OTH_imp_scal)[:,1]


		else:
			if 'OTH' in flatPriors:
				logging.info("Using flat priors for OTH")
				y_OTH_pred = np.full(len(x_OTH), flatPriors['OTH'])
			else:
				logging.info("Ignoring all OTH")
				y_OTH_pred = np.full(len(x_OTH), np.nan)


	logging.info(" ")





	# combine and output to file
	logging.info("Outputting the priors to file")


	if args.cnv:
		prior_prob = pd.DataFrame()
		prior_prob["ID"] = np.concatenate((y_DEL["ID"], y_DUP["ID"]))
		prior_prob["prior"] = np.concatenate((y_DEL_pred, y_DUP_pred))


		x_DEL_data = pd.DataFrame()		
		x_DUP_data = pd.DataFrame()		


		# get the regression input data if available
		if os.path.isfile(modelPrefix + '.DEL_predictors.npy'):
			#x_DEL_data = pd.DataFrame(x_DEL_imp_scal, index = x_DEL_index, columns = x_DEL.columns)
			x_DEL_data = pd.DataFrame(x_DEL, index = x_DEL_index, columns = x_DEL.columns)
		
		else:
			if len(x_DEL_index) > 0: 
				x_DEL_data = pd.DataFrame(np.nan, index=range(len(x_DEL_index)), columns = x_DEL.columns)



		if os.path.isfile(modelPrefix + '.DUP_predictors.npy'):
			#x_DUP_data = pd.DataFrame(x_DUP_imp_scal, index = x_DUP_index, columns = x_DUP.columns)
			x_DUP_data = pd.DataFrame(x_DUP, index = x_DUP_index, columns = x_DUP.columns)

		else:
			if len(x_DUP_index) > 0:
				x_DUP_data = pd.DataFrame(np.nan, index=range(len(x_DUP_index)), columns = x_DUP.columns)
		


		x_data = pd.concat([x_DEL_data, x_DUP_data], ignore_index=True, sort=False)
		combined = pd.concat([x_data, prior_prob], axis=1)
		df_flat = pd.DataFrame(flat_DATA, columns = ['ID', 'prior'])
		merged = pd.concat([combined, df_flat], sort = False)
		merged["PriorOC"] = merged["prior"] / (1 - merged["prior"])
		merged["logPriorOC"] = np.log10(merged["PriorOC"])
		
		orderFirst = [ "ID", "prior", "PriorOC", "logPriorOC" ]
		orderSecond = [ x for x in merged.columns.tolist() if x not in orderFirst ]
		merged = merged[ orderFirst + orderSecond ]


		merged.to_csv(args.outputDir + outputPrefix+".priors.txt", index=False, sep='\t', na_rep='.')



	else:
		prior_prob = pd.DataFrame()
		prior_prob["ID"] = np.concatenate((y_IND["ID"], y_MIS["ID"], y_OTH["ID"]))
		prior_prob["csq"] = np.concatenate((x_IND_csq, x_MIS_csq, x_OTH_csq))
		prior_prob["impact"] = np.concatenate((x_IND_impact, x_MIS_impact, x_OTH_impact))
		prior_prob["Gene"] = np.concatenate((y_IND["Gene"], y_MIS["Gene"], y_OTH["Gene"]))
		prior_prob["prior"] = np.concatenate((y_IND_pred, y_MIS_pred, y_OTH_pred))


		x_IND_data = pd.DataFrame()		
		x_MIS_data = pd.DataFrame()		
		x_OTH_data = pd.DataFrame()		


		# get the regression input data if available
		if os.path.isfile(modelPrefix + '.IND_predictors.npy'):
			x_IND_data = pd.DataFrame(x_IND, index = x_IND_index, columns = x_IND.columns)
		
		else:
			if len(x_IND_index) > 0: 
				x_IND_data = pd.DataFrame(np.nan, index=range(len(x_IND_index)), columns = x_IND.columns)



		if os.path.isfile(modelPrefix + '.MIS_predictors.npy'):
			x_MIS_data = pd.DataFrame(x_MIS, index = x_MIS_index, columns = x_MIS.columns)

		else:
			if len(x_MIS_index) > 0:
				x_MIS_data = pd.DataFrame(np.nan, index=range(len(x_MIS_index)), columns = x_MIS.columns)
		


		if os.path.isfile(modelPrefix + '.OTH_predictors.npy'):
			x_OTH_data = pd.DataFrame(x_OTH, index = x_OTH_index, columns = x_OTH.columns)
		
		else:
			if len(x_OTH_index) > 0:
				x_OTH_data = pd.DataFrame(np.nan, index=range(len(x_OTH_index)), columns = x_OTH.columns)



		x_data = pd.concat([x_IND_data, x_MIS_data, x_OTH_data], ignore_index=True, sort=False)
		combined = pd.concat([x_data, prior_prob], axis=1)
		df_flat = pd.DataFrame(flat_DATA, columns = ['ID', 'prior'])
		merged = pd.concat([combined, df_flat], sort = False)
		merged["PriorOC"] = merged["prior"] / (1 - merged["prior"])
		merged["logPriorOC"] = np.log10(merged["PriorOC"])
		
		merged = merged[merged.columns.drop(list(merged.filter(regex='csqVEP_')))]
		merged = merged[merged.columns.drop(list(merged.filter(regex='typeVEP_')))]
		merged = merged[merged.columns.drop(list(merged.filter(regex='impactVEP_')))]

		orderFirst = [ "ID", "Gene", "csq", "impact", "prior", "PriorOC", "logPriorOC" ]
		orderSecond = [ x for x in merged.columns.tolist() if x not in orderFirst ]
		merged = merged[ orderFirst + orderSecond ]


		merged.to_csv(args.outputDir + outputPrefix+".priors.txt", index=False, sep='\t', na_rep='.')


	# get IDs of variants with a prior
	merged_ID = merged[merged["prior"].notna()]["ID"].unique()
	np.save(args.tempDir + outputPrefix+".priors.ID.npy", merged_ID)



	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")

	







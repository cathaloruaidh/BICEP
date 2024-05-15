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

		with open(predictorsFile, 'r') as f:
			for line in f:
				(key, value)=line.split()
				d[key] = value

		keysPredictors = sorted(list(d.keys()))
		keysDescPred = sorted([ x for x in keysPredictors if d[x] == "-" ] + [alleleFrequency])
		keysAscPred  = sorted([ x for x in keysPredictors if d[x] == "+" ])
		keysPredictors = sorted(list(d.keys()) + [alleleFrequency])


	else:
		keysPredictors = sorted([ "CADD_PHRED", "FATHMM_score", "MPC_score", "Polyphen2_HDIV_score", "REVEL_score", "SIFT_score" ] + [alleleFrequency])
		keysDescPred = sorted([ "SIFT_score", "FATHMM_score" ] + [alleleFrequency])
		keysAscPred  = sorted([ x for x in keysPredictors if x not in keysDescPred ])
	



	logging.info("Converting annotation data to list")

	# save variants in list 
	DATA = []
	flat_DATA = []
	CHROMS = set([ "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22" ])

	for variant in vcf:
		# remove variants not on autosomes
		if len( set([variant.CHROM, "chr"+variant.CHROM]) & CHROMS ) == 0:
			continue

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
			dictVEP = dict(zip(keys, tmp.T))


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
				#dictVEP["Consequence_select"] = np.nan
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
				l = [ ID, csqVEP[i]["SYMBOL"], typeVEP, csqVEP[i]["Consequence_select"] ]

				for key in keysAscPred + keysDescPred:
					l.append(csqVEP[i][key])

				DATA.append(l) 


		# otherwise flat, nonCoding prior
		else:
			if 'nonCoding' in flatPriors.keys():
				flat_DATA.append([ID, flatPriors['nonCoding']])

			else:
				flat_DATA.append([ID, np.nan])


	logging.info(" ")



	################################################################################
	# Apply the regression prediction
	################################################################################

	pd.set_option('display.max_colwidth', 100)

	# read in the data
	df = pd.DataFrame(DATA, columns = ['ID', 'Gene', 'typeVEP', 'csqVEP'] + keysAscPred + keysDescPred )


	df['csqVEP'] = pd.Categorical(df['csqVEP'], categories = sorted(vepCSQRank.keys()))
	df['typeVEP'] = pd.Categorical(df['typeVEP'], categories = ['indel', 'SNV'])


	# set missing allele frequencies to zero
	df[alleleFrequency] = df[alleleFrequency].fillna(0.0)


	x = df.filter( ['csqVEP', 'typeVEP'] + keysAscPred + keysDescPred)
	


	## INDELS
	logging.info("INDELS")

	x_indel = x[ x['typeVEP'] == 'indel' ]
	x_indel_csq = x_indel['csqVEP']
	x_indel = pd.get_dummies(x_indel, columns = ['csqVEP'])
	x_indel_index = x_indel.index

	y_indel = df[ df['typeVEP'] == 'indel' ]



	if os.path.isfile(modelPrefix + '.predictors_indel.npy'):

		# load in the predictor IDs for the models
		with open(modelPrefix + '.predictors_indel.npy', 'rb') as f:
			predictors_indel = np.load(f, allow_pickle = True)
			predictors_indel = [ s.replace('CV', 'VEP') for s in predictors_indel ]

		x_indel = x_indel[predictors_indel]



		# impute the missing data
		logging.info("Impute the data")

		with open(modelPrefix + '.imp_indel.pkl', 'rb') as f:
			imp_indel = pickle.load(f)

		x_indel_imp = imp_indel.transform(x_indel)



		# scale the data
		logging.info("Scaling to [0,1]")

		with open(modelPrefix + '.scal_indel.pkl', 'rb') as f:
			scal_indel = pickle.load(f)

		x_indel_imp_scal = scal_indel.transform(x_indel_imp)


		# run logistic regression
		logging.info("Apply logistic regression")

		with open(modelPrefix + '.logReg_indel.pkl', 'rb') as f:
			logReg_indel = pickle.load(f)


		y_indel_pred = logReg_indel.predict_proba(x_indel_imp_scal)[:,1]


	else:
		if 'indel' in flatPriors.keys():
			logging.info("Using flat priors for indels. ")
			y_indel_pred = np.full(len(x_indel), flatPriors['indel'])
		else:
			logging.info("Ignoring all indels. ")
			y_indel_pred = np.full(len(x_indel), np.nan)


	logging.info(" ")



	## MISSENSE SNV
	logging.info("MISSENSE SNV")

	x_missense = x[ x['csqVEP'] == 'missense_variant' ]
	x_missense_csq = x_missense['csqVEP']
	x_missense_index = x_missense.index


	y_missense = df[ df['csqVEP'] == 'missense_variant' ]

	if os.path.isfile(modelPrefix + '.predictors_missense.npy'):

		# load in the predictor IDs for the models
		with open(modelPrefix + '.predictors_missense.npy', 'rb') as f:
			predictors_missense = np.load(f, allow_pickle = True)
			predictors_missense = [ s.replace('CV', 'VEP') for s in predictors_missense ]

		x_missense = x_missense[predictors_missense]


		# impute the missing data
		logging.info("Impute the data")

		with open(modelPrefix + '.imp_missense.pkl', 'rb') as f:
			imp_missense = pickle.load(f)


		x_missense_imp = imp_missense.transform(x_missense)
		x_missense_imp_df = pd.DataFrame(x_missense_imp, columns = x_missense.columns)


		# scale the data
		logging.info("Scaling to [0,1]")

		with open(modelPrefix + '.scal_missense.pkl', 'rb') as f:
			scal_missense = pickle.load(f)

		x_missense_imp_scal = scal_missense.transform(x_missense_imp)

		x_missense_imp_scal_df = pd.DataFrame(x_missense_imp_scal, columns = x_missense.columns)

		# run logistic regression
		logging.info("Apply logistic regression")

		with open(modelPrefix + '.logReg_missense.pkl', 'rb') as f:
			logReg_missense = pickle.load(f)

		y_missense_pred = logReg_missense.predict_proba(x_missense_imp_scal)[:,1]

	
	else:
		if 'missense' in flatPriors.keys():
			logging.info("Using flat priors for missense variants. ")
			y_missense_pred = np.full(len(x_missense), flatPriors['missense'])
		else:
			logging.info("Ignoring all missense variants")
			y_missense_pred = np.full(len(x_missense), np.nan)

	logging.info(" ")



	## NON-MISSENSE SNV
	logging.info("NON-MISSENSE SNV")

	x_nonMissenseSNV = x[ (x['csqVEP'] != 'missense_variant') & (x['typeVEP'] == 'SNV') ]
	x_nonMissenseSNV_csq = x_nonMissenseSNV['csqVEP']
	x_nonMissenseSNV = pd.get_dummies(x_nonMissenseSNV, columns = ['csqVEP'])
	x_nonMissenseSNV_index = x_nonMissenseSNV.index

	y_nonMissenseSNV = df[ (df['csqVEP'] != 'missense_variant') & (df['typeVEP'] == 'SNV') ]

		
	if os.path.isfile(modelPrefix + '.predictors_nonMissenseSNV.npy'):

		# load in the predictor IDs for the models
		with open(modelPrefix + '.predictors_nonMissenseSNV.npy', 'rb') as f:
			predictors_nonMissenseSNV = np.load(f, allow_pickle = True)
			predictors_nonMissenseSNV = [ s.replace('CV', 'VEP') for s in predictors_nonMissenseSNV ]

		x_nonMissenseSNV = x_nonMissenseSNV[predictors_nonMissenseSNV]


		# impute the missing data
		logging.info("Impute the data")

		with open(modelPrefix + '.imp_nonMissenseSNV.pkl', 'rb') as f:
			imp_nonMissenseSNV = pickle.load(f)
		
		x_nonMissenseSNV_imp = imp_nonMissenseSNV.transform(x_nonMissenseSNV)



		# scale the data
		logging.info("Scaling to [0,1]")

		with open(modelPrefix + '.scal_nonMissenseSNV.pkl', 'rb') as f:
			scal_nonMissenseSNV = pickle.load(f)

		x_nonMissenseSNV_imp_scal = scal_nonMissenseSNV.transform(x_nonMissenseSNV_imp)



		# run logistic regression
		logging.info("Apply logistic regression")

		with open(modelPrefix + '.logReg_nonMissenseSNV.pkl', 'rb') as f:
			logReg_nonMissenseSNV = pickle.load(f)


		y_nonMissenseSNV_pred = logReg_nonMissenseSNV.predict_proba(x_nonMissenseSNV_imp_scal)[:,1]


	else:
		if 'nonMissenseSNV' in flatPriors:
			logging.info("Using flat priors for non-missense SNVs")
			y_nonMissenseSNV_pred = np.full(len(x_nonMissenseSNV), flatPriors['nonMissenseSNV'])
		else:
			logging.info("Ignoring all non-missense SNVs")
			y_nonMissenseSNV_pred = np.full(len(x_nonMissenseSNV), np.nan)


	logging.info(" ")





	# combine and output to file
	logging.info("Outputting the priors to file")

	prior_prob = pd.DataFrame()
	prior_prob["ID"] = np.concatenate((y_indel["ID"], y_missense["ID"], y_nonMissenseSNV["ID"]))
	prior_prob["csq"] = np.concatenate((x_indel_csq, x_missense_csq, x_nonMissenseSNV_csq))
	prior_prob["Gene"] = np.concatenate((y_indel["Gene"], y_missense["Gene"], y_nonMissenseSNV["Gene"]))
	prior_prob["prior"] = np.concatenate((y_indel_pred, y_missense_pred, y_nonMissenseSNV_pred))


	x_indel_data = pd.DataFrame()		
	x_missense_data = pd.DataFrame()		
	x_nonMissenseSNV_data = pd.DataFrame()		


	# get the regression input data if available
	if os.path.isfile(modelPrefix + '.predictors_indel.npy'):
		#x_indel_data = pd.DataFrame(x_indel_imp_scal, index = x_indel_index, columns = x_indel.columns)
		x_indel_data = pd.DataFrame(x_indel, index = x_indel_index, columns = x_indel.columns)
	
	else:
		if len(x_indel_index) > 0: 
			x_indel_data = pd.DataFrame(np.nan, index=range(len(x_indel_index)), columns = x_indel.columns)



	if os.path.isfile(modelPrefix + '.predictors_missense.npy'):
		#x_missense_data = pd.DataFrame(x_missense_imp_scal, index = x_missense_index, columns = x_missense.columns)
		x_missense_data = pd.DataFrame(x_missense, index = x_missense_index, columns = x_missense.columns)

	else:
		if len(x_missense_index) > 0:
			x_missense_data = pd.DataFrame(np.nan, index=range(len(x_missense_index)), columns = x_missense.columns)
	


	if os.path.isfile(modelPrefix + '.predictors_nonMissenseSNV.npy'):
		#x_nonMissenseSNV_data = pd.DataFrame(x_nonMissenseSNV_imp_scal, index = x_nonMissenseSNV_index, columns = x_nonMissenseSNV.columns)
		x_nonMissenseSNV_data = pd.DataFrame(x_nonMissenseSNV, index = x_nonMissenseSNV_index, columns = x_nonMissenseSNV.columns)
	
	else:
		if len(x_nonMissenseSNV_index) > 0:
			x_nonMissenseSNV_data = pd.DataFrame(np.nan, index=range(len(x_nonMissenseSNV_index)), columns = x_nonMissenseSNV.columns)



	x_data = pd.concat([x_indel_data, x_missense_data, x_nonMissenseSNV_data], ignore_index=True, sort=False)
	combined = pd.concat([x_data, prior_prob], axis=1)
	df_flat = pd.DataFrame(flat_DATA, columns = ['ID', 'prior'])
	merged = pd.concat([combined, df_flat], sort = False)
	merged["PriorOC"] = merged["prior"] / (1 - merged["prior"])
	merged["logPriorOC"] = np.log10(merged["PriorOC"])
	
	merged = merged[merged.columns.drop(list(merged.filter(regex='csqVEP_')))]
	merged = merged[merged.columns.drop(list(merged.filter(regex='typeVEP_')))]


	merged.to_csv(args.outputDir + outputPrefix+".priors.txt", index=False, sep='\t', na_rep='.')


	# get IDs of variants with a prior
	merged_ID = merged[merged["prior"].notna()]["ID"].unique()
	np.save(args.tempDir + outputPrefix+".priors.ID.npy", merged_ID)



	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")

	







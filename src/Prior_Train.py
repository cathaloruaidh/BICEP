import cProfile
import logging
import math
import os
import pickle
import pprint
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


from cyvcf2 import VCF, Writer
from joblib import dump, load
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, r2_score
from sklearn.preprocessing import MinMaxScaler








# main function
def PT_main(args):

	logging.info("PRIOR - TRAIN")
	logging.info(" ")


	# ignore warnings
	warnings.filterwarnings("ignore")
	pd.set_option('display.max_columns', None)



	# command line arguments
	nCores = 1
	excludeFile = None
	includeFile = None
	clinVarFullVcfFile = None
	clinVarAnnoVcfFile = None
	outputPrefix = None
	logLevel = None
	benignFile = None
	pathogenicFile = None
	predictorsFile = None

	if args.clinvar is not None:
		clinVarAnnoVcfFile = args.clinvar
		clinVarFullVcfFile = args.clinvar

	if args.exclude is not None:
		excludeFile = args.exclude

	if args.include is not None:
		includeFile = args.include

	if args.benign is not None:
		benignFile = args.benign

	if args.prefix is not None:
		outputPrefix = args.prefix

	if args.pathogenic is not None:
		pathogenicFile = args.pathogenic

	if args.predictors is not None:
		predictorsFile = args.predictors




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










	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")


	# get VCF file and variant/sample information
	CV_full_vcf = VCF(clinVarFullVcfFile, gts012=True)


	# generate the flat priors
	logging.info("Generating the flat priors")

	ALL_DATA = []
	for variant in CV_full_vcf:
		sig = variant.INFO.get('CLNSIG')
		if sig is not None:
			sig = sig.split(",", 1)[0]

		gene = variant.INFO.get('GENEINFO')
		vc = variant.INFO.get('CLNVC')
		mc = variant.INFO.get('MC')

		if not (vc == "single_nucleotide_variant" or vc == "Indel"):
			continue
		ALL_DATA.append([ sig, vc, gene, mc ])
	
	df_all = pd.DataFrame(ALL_DATA, columns=['sig', 'vc', 'gene', 'mc'])
	df_all_path = df_all.dropna(subset=['sig'])
	df_all_path = df_all_path[ df_all_path['sig'].str.contains("athogenic") ]
	df_all_path = df_all_path[ df_all_path['sig'].str.contains("Conflicting") == False ]


	flatPrior_nonCoding =  df_all_path['gene'].isna().sum() / df_all['gene'].isna().sum()


	df_all = df_all.dropna(subset=['gene'])
	df_all_path = df_all_path.dropna(subset=['gene'])

	flatPrior_indel = len(df_all_path[ df_all_path['vc'] == "Indel" ].index) / len(df_all[ df_all['vc'] == "Indel" ].index)


	df_all = df_all.dropna(subset=['mc'])
	df_all_path = df_all_path.dropna(subset=['mc'])

	flatPrior_missense = len(df_all_path[ df_all_path['mc'].str.contains("missense") ].index) / len(df_all[ df_all['mc'].str.contains("missense") ].index)

	df_all = df_all[ df_all['vc'] == "single_nucleotide_variant" ]
	df_all_path = df_all_path[ df_all_path['vc'] == "single_nucleotide_variant" ]

	flatPrior_nonMissenseSNV = len(df_all_path[ ~df_all_path['mc'].str.contains("missense") ].index) / len(df_all[ ~df_all['mc'].str.contains("missense") ].index)


	flatPriors = { 'nonCoding' : flatPrior_nonCoding, 'indel' : flatPrior_indel, 'missense' : flatPrior_missense, 'nonMissenseSNV' : flatPrior_nonMissenseSNV }



	with open(args.tempDir + outputPrefix+'.flatPriors.pkl', 'wb') as f:
		pickle.dump(flatPriors, f)





	# now, use annotated file 
	logging.info("Parsing the annotation information")
	CV_anno_vcf = VCF(clinVarAnnoVcfFile, gts012=True)

	keys = re.sub('^.*?: ', '', CV_anno_vcf.get_header_type('CSQ')['Description']).split("|")
	keys = [ key.strip("\"") for key in keys ]


	if predictorsFile is not None:

		d = {}

		with open(predictorsFile, 'r') as f:
			for line in f:
				(key, value)=line.split()
				d[key] = value


		keysPredictors = sorted(list(d.keys()))
		keysDescPred = sorted([ x for x in keysPredictors if d[x] == "-" ])
		keysAscPred  = sorted([ x for x in keysPredictors if d[x] == "+" ])


	else:
		keysPredictors = sorted([ "CADD_PHRED", "FATHMM_score", "MPC_score", "Polyphen2_HDIV_score", "REVEL_score", "SIFT_score", "gnomAD_v2_exome_AF_popmax" ])
		keysDescPred = sorted([ "FATHMM_score", "SIFT_score", "gnomAD_v2_exome_AF_popmax" ])
		keysAscPred  = sorted([ x for x in keysPredictors if x not in keysDescPred ])




	# save variants in list 
	DATA = []
	for variant in CV_anno_vcf:
		# get the ID of the variant
		ID = variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0] 
		
		msg = "Reading variant " + ID
		logging.debug(msg)



		# get clinvar allele ID
		alleleID = variant.ID




		# get the clinvar significance, ignoring anything after a comma
		sigCV = variant.INFO.get('CLNSIG').split(",", 1)[0]



		# get the short significance
		if (sigCV == "Benign") or (sigCV == "Benign/Likely_benign") or (sigCV == "Likely_benign"):
			setCV = "BENIGN"
		elif (sigCV == "Likely_pathogenic") or (sigCV == "Pathogenic/Likely_pathogenic") or (sigCV == "Pathogenic"):
			setCV = "PATHOGENIC"


		# get the variant type, ignore if not SNV or indel
		if variant.INFO.get('CLNVC') == "single_nucleotide_variant":
			typeCV = "SNV"
		elif variant.INFO.get('CLNVC') == "Indel":
			typeCV = "indel"
		else:
			continue



		# get the clinvar gene list
		tmp = variant.INFO.get('GENEINFO')
		if (tmp is None) or ("|" in tmp):
			continue
		geneCV = re.sub(':.*?$', '', tmp)



		# get the variant clinvar consequence
		tmp = variant.INFO.get('MC')	
		if (tmp is None) or ("," in tmp):
			continue
		csqCV = re.sub('^.*?\|', '', tmp)


		# change 'nonsense' to 'stop_gained'
		csqCV = re.sub('nonsense', 'stop_gained', csqCV)



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
			

			# extract the transcript-specific metrics from dbNSFP
			# and pick the most deleterious value
			for key in keysAscPred:
				l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
				if len(l) == 0:
					dictVEP[key] = "."
				else:
					dictVEP[key] = max(l)

			for key in keysDescPred:
				l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
				if len(l) == 0:
					dictVEP[key] = "."
				else:
					dictVEP[key] = min(l)

			
			# if the ClinVar consequence is not in the VEP consequences, do not include
			# the transcript
			incl = False
			if csqCV in dictVEP['Consequence_split']:
				incl = True
			
			if incl == False:
				add = False


			# if the VEP gene is not the same as the ClinVar gene, do not include the
			# transcript
			if dictVEP['SYMBOL'] != geneCV:
				add = False


			# if the transcript is not protein-coding reject it
			if dictVEP['BIOTYPE'] != "protein_coding":
				add = False

			
			# if the transcript hasn't been rejected, add it to the list
			if add:
				csqVEP.append(dictVEP)




		# summarize scores across all transcripts
		if len(csqVEP) > 0:
			l = [ ID, setCV, geneCV, csqCV, typeCV, alleleID ]

	
			for key in keysAscPred:
				m = csqVEP[0][key]
				for i in range(len(csqVEP)):
					if csqVEP[i][key] == ".":
						continue
					if m == ".":
						m = csqVEP[i][key]
						continue
					if csqVEP[i][key] > m:
						m = csqVEP[i][key]
				l.append(m)


			for key in keysDescPred:
				m = csqVEP[0][key]
				for i in range(len(csqVEP)):
					if csqVEP[i][key] == ".":
						continue
					if m == ".":
						m = csqVEP[i][key]
						continue
					if csqVEP[i][key] < m:
						m = csqVEP[i][key]
				l.append(m)

			DATA.append(l) 


		# otherwise print to error file
		#else:
		#	with open(outputPrefix + ".err", 'a') as f:
		#		print(ID, "\t", setCV, "\t", csqCV, file=f)

	
	for i in range(len(DATA)):
		for j in range(5, len(DATA[i])):
			if DATA[i][j] == "." or DATA[i][j] == "":
				DATA[i][j] = np.nan
			else:
				DATA[i][j] = float(DATA[i][j])
			



	################################################################################
	# Run the Regression model
	################################################################################


	# read in the data
	logging.debug("Converting to DataFrame")
	#df = pd.DataFrame(DATA, columns = [ 'ID', 'setCV', 'geneCV', 'csqCV', 'typeCV', 'alleleID' ] + )
	df = pd.DataFrame(DATA, columns = [ 'ID', 'setCV', 'geneCV', 'csqCV', 'typeCV', 'alleleID' ] + keysAscPred + keysDescPred )
	df['alleleID'] = df.alleleID.astype(str).replace('\.0', '', regex=True)

	df.to_csv("DATA.txt", index=False, sep='\t', na_rep='.')



	# if there are pathogenic/benign files, subset to these
	if (pathogenicFile is not None) and (benignFile is not None):
		logging.info("Setting the pathogenic/benign training elements from input file")

		df['setCV'] = "."

		with open(pathogenicFile, 'r') as f:
			pathogenicList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df.loc[df.alleleID.isin(pathogenicList.alleleID), 'setCV'] = "PATHOGENIC"

		with open(benignFile, 'r') as f:
			benignList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df.loc[df.alleleID.isin(benignList.alleleID), 'setCV'] = "BENIGN"

		df = df.drop(df[df['setCV'] == "."].index)



	if pathogenicFile is not None and benignFile is None:
		logging.warning("Pathogenic file is not specified, ignoring benign file")
	
	
	if pathogenicFile is None and benignFile is not None:
		logging.warning("Benign file is not specified, ignoring pathogenic file")




	# if there is an exclude file, remove them 
	if excludeFile is not None:
		logging.info("Removing variants in exclude file")
		with open(excludeFile, 'r') as f:
			excludeList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df = df[~df.alleleID.isin(excludeList.alleleID)]


	# if there is an include file, remove everything else 
	if includeFile is not None:
		logging.info("Susbetting to variants in include file")
		with open(includeFile, 'r') as f:
			includeList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df = df[df.alleleID.isin(includeList.alleleID)]


	

	
	msg = "A total of " + str(len(df.index)) + " variants used (" + str((df.setCV.values == 'PATHOGENIC').sum()) + " PATH and " + str((df.setCV.values == 'BENIGN').sum()) + " BEN)"

	logging.info(msg)





	# subset to variables being used
	#x = df.filter(['ID', 'alleleID', 'geneCV'] + keysPredictors + ['csqCV', 'typeCV'])
	x = df.filter([ 'ID', 'geneCV', 'csqCV', 'typeCV', 'alleleID' ] + keysAscPred + keysDescPred)


	y = df.filter(['setCV', 'csqCV', 'typeCV'])


	logging.info(" ")


	## INDELS
	logging.info("INDELS (CADD + AF + CSQ)")
	x_indel = x[ x['typeCV'] == 'indel' ]
	ID_indel = x_indel['ID']
	alleleID_indel = x_indel['alleleID']
	geneCV_indel = x_indel['geneCV']
	x_indel = x_indel.drop(['ID', 'alleleID', 'geneCV'], axis=1, errors='ignore')
	x_indel_index = x_indel.index
	x_indel_csq = x_indel['csqCV'] 
	x_indel['csqCV'] = pd.Categorical(x_indel['csqCV'], categories = sorted(x_indel['csqCV'].unique()))
	csqCV_indel = [ x for x in x_indel['csqCV'] if x in vepCSQRank.keys() ]
	d_indel = dict((k, vepCSQRank[k]) for k in csqCV_indel)
	drop_indel = "csqCV_" + max(d_indel, key=d_indel.get)
	x_indel = pd.get_dummies(x_indel, columns = ['csqCV']).drop(drop_indel, axis=1)


	y_indel = y[ y['typeCV'] == 'indel' ]
	y_indel = y_indel.drop(['csqCV', 'typeCV'], axis=1, errors='ignore')
	y_indel = y_indel.values.reshape(-1,1)


	uniq, counts = np.unique(y_indel, return_counts = True)

	if (len(x_indel.index) > 0) and (len(uniq) == 2) and (counts.min() > 15*len(x_indel.columns)):
		x_indel = x_indel.drop(['FATHMM_score', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'typeCV'], axis=1, errors='ignore')


		msg = "A total of " + str(np.size(y_indel)) + " indels used (" + str(sum(y_indel == 'PATHOGENIC')) + " PATH and " + str(sum(y_indel == 'BENIGN')) + " BEN)"
		logging.info(msg)
	
		# remove columns that are all NA
		x_indel = x_indel.dropna(axis=1, how='all')

	
		# set missing allele frequencies to zero
		if "gnomAD_v2_exome_AF_popmax" in x_indel.columns:
			x_indel["gnomAD_v2_exome_AF_popmax"] = x_indel["gnomAD_v2_exome_AF_popmax"].fillna(0.0)



		# impute the missing data
		logging.info("Impute the data")
		imp_indel = SimpleImputer(strategy = 'median')
		imp_indel.fit(x_indel)

		imp_indel.feature_names = list(x_indel.columns.values)
		x_indel_imp = imp_indel.transform(x_indel)

		with open(args.tempDir + outputPrefix+'.imp_indel.pkl', 'wb') as f:
			pickle.dump(imp_indel, f)

		
		# scale the data
		logging.info("Scaling to [0,1]")
		scal_indel = MinMaxScaler(clip = 'true')
		scal_indel.fit(x_indel_imp)

		scal_indel.feature_names = list(x_indel.columns.values)
		x_indel_imp_scal = scal_indel.transform(x_indel_imp)

		with open(args.tempDir + outputPrefix+'.scal_indel.pkl', 'wb') as f:
			pickle.dump(scal_indel, f)



		# run logistic regression
		logging.info("Run logistic regression")
		logReg_indel = LogisticRegression(penalty = 'none')
		logReg_indel.fit(x_indel_imp_scal, y_indel)

		logReg_indel.feature_names = list(x_indel.columns.values)

		with open(args.tempDir + outputPrefix+'.logReg_indel.pkl', 'wb') as f:
			pickle.dump(logReg_indel, f)

		with open(args.tempDir + outputPrefix+'.predictors_indel.npy', 'wb') as f:
			np.save(f, x_indel.columns)

		results_indel = logReg_indel.predict_proba(x_indel_imp_scal)[:,1]
		
		with open(args.tempDir + outputPrefix + ".indel_coef.txt", 'a') as f:
			pprint.pprint(list(zip(logReg_indel.feature_names, np.round(logReg_indel.coef_.flatten(), 6))), f)
			print("Intercept: ", np.round(logReg_indel.intercept_, 6), file=f)
	else:
		msg = "Not enough indels in training data, using flat prior: " + str(np.round(flatPriors['indel'], 6))
		logging.info(msg)
		results_indel = np.full(len(x_indel.index), flatPriors['indel'])

	logging.info(" ")
	logging.info(" ")



	## MISSENSE SNV
	logging.info("MISSENSE SNV (AF + 5_PRED)")
	x_missense = x[ (x['csqCV'] == 'missense_variant') & (x['typeCV'] == 'SNV') ]
	x_missense_csq = x_missense['csqCV'] 
	x_missense = x_missense.drop(['CADD_PHRED', 'typeCV', 'csqCV'], axis=1, errors='ignore')



	y_missense = y[ (y['csqCV'] == 'missense_variant') & (y['typeCV'] == 'SNV') ]
	y_missense = y_missense.drop(['csqCV', 'typeCV'], axis=1)
	y_missense = y_missense.values.reshape(-1,1)


	# get IDs
	ID_missense = x_missense['ID']
	alleleID_missense = x_missense['alleleID']
	geneCV_missense = x_missense['geneCV']
	x_missense = x_missense.drop(['ID', 'alleleID', 'geneCV'], axis=1)
	x_missense_index = x_missense.index


	uniq, counts = np.unique(y_missense, return_counts = True)

	if (len(x_missense.index) > 0) and (len(uniq) == 2) and (counts.min() > 15*len(x_missense.columns)):
		msg = "A total of " + str(np.size(y_missense)) + " missense variants used (" + str(sum(y_missense == 'PATHOGENIC')) + " PATH and " + str(sum(y_missense == 'BENIGN')) + " BEN)"
		logging.info(msg)
		
		# remove columns that are all NA
		x_missense = x_missense.dropna(axis=1, how='all')

	
		# set missing allele frequencies to zero
		if "gnomAD_v2_exome_AF_popmax" in x_missense.columns:
			x_missense["gnomAD_v2_exome_AF_popmax"] = x_missense["gnomAD_v2_exome_AF_popmax"].fillna(0.0)


		# impute the missing data
		logging.info("Impute the data")
		imp_missense = SimpleImputer(strategy = 'median')
		imp_missense.fit(x_missense)

		imp_missense.feature_names = list(x_missense.columns.values)

		x_missense_imp = imp_missense.transform(x_missense)

		with open(args.tempDir + outputPrefix+'.imp_missense.pkl', 'wb') as f:
			pickle.dump(imp_missense, f)


		# scale the data
		logging.info("Scaling to [0,1]")
		scal_missense = MinMaxScaler(clip = 'true')
		scal_missense.fit(x_missense_imp)

		scal_missense.feature_names = list(x_missense.columns.values)

		x_missense_imp_scal = scal_missense.transform(x_missense_imp)

		with open(args.tempDir + outputPrefix+'.scal_missense.pkl', 'wb') as f:
			pickle.dump(scal_missense, f)



		# run logistic regression
		logging.info("Run logistic regression")
		logReg_missense = LogisticRegression(penalty = 'none')
		logReg_missense.fit(x_missense_imp_scal, y_missense)

		logReg_missense.feature_names = list(x_missense.columns.values)

		with open(args.tempDir + outputPrefix+'.logReg_missense.pkl', 'wb') as f:
			pickle.dump(logReg_missense, f)

		with open(args.tempDir + outputPrefix+'.predictors_missense.npy', 'wb') as f:
			np.save(f, x_missense.columns)

		results_missense = logReg_missense.predict_proba(x_missense_imp_scal)[:,1]

		with open(args.tempDir + outputPrefix + ".missense_coef.txt", 'a') as f:
		 pprint.pprint(list(zip(logReg_missense.feature_names, np.round(logReg_missense.coef_.flatten(), 6))), f)
		 print("Intercept: ", np.round(logReg_missense.intercept_, 6), file=f)
	else:
		msg = "Not enough missense variants in training data, using flat prior: " + str(np.round(flatPriors['missense'], 6))
		logging.info(msg)
		results_missense = np.full(len(x_missense.index), flatPriors['missense'])


	logging.info(" ")
	logging.info(" ")



	## NON-MISSENSE SNV
	logging.info("NON-MISSENSE SNV (CADD + AF + CSQ)")
	x_nonMissenseSNV = x[ (x['csqCV'] != 'missense_variant') & (x['typeCV'] == 'SNV') ]
	x_nonMissenseSNV_csq = x_nonMissenseSNV['csqCV'] 


	y_nonMissenseSNV = y[ (y['csqCV'] != 'missense_variant') & (y['typeCV'] == 'SNV') ]
	y_nonMissenseSNV = y_nonMissenseSNV.drop(['csqCV', 'typeCV'], axis=1)
	y_nonMissenseSNV = y_nonMissenseSNV.values.reshape(-1,1)


	# get IDs
	ID_nonMissenseSNV = x_nonMissenseSNV['ID']
	alleleID_nonMissenseSNV = x_nonMissenseSNV['alleleID']
	geneCV_nonMissenseSNV = x_nonMissenseSNV['geneCV']
	x_nonMissenseSNV = x_nonMissenseSNV.drop(['ID', 'alleleID', 'geneCV'], axis=1)
	x_nonMissenseSNV_index = x_nonMissenseSNV.index

	x_nonMissenseSNV['csqCV'] = pd.Categorical(x_nonMissenseSNV['csqCV'], categories = sorted(x_nonMissenseSNV['csqCV'].unique()))
	csqCV_nonMissenseSNV = [ x for x in x_nonMissenseSNV['csqCV'] if x in vepCSQRank.keys() ]
	d_nonMissenseSNV = dict((k, vepCSQRank[k]) for k in csqCV_nonMissenseSNV)
	drop_nonMissenseSNV = "csqCV_" + max(d_nonMissenseSNV, key=d_nonMissenseSNV.get)
	x_nonMissenseSNV = pd.get_dummies(x_nonMissenseSNV, columns = ['csqCV']).drop(drop_nonMissenseSNV, axis=1)



	uniq, counts = np.unique(y_nonMissenseSNV, return_counts = True)

	if (len(x_nonMissenseSNV.index) > 0) and (len(uniq) == 2) and (counts.min() > 15*len(x_nonMissenseSNV.columns)):
		x_nonMissenseSNV = x_nonMissenseSNV.drop(['FATHMM_score', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'typeCV'], axis=1, errors='ignore')


		msg = "A total of " + str(np.size(y_nonMissenseSNV)) + " non-missense SNVs used (" + str(sum(y_nonMissenseSNV == 'PATHOGENIC')) + " PATH and " + str(sum(y_nonMissenseSNV == 'BENIGN')) + " BEN)"
		logging.info(msg)
		
		# remove columns that are all NA
		x_nonMissenseSNV = x_nonMissenseSNV.dropna(axis=1, how='all')

	
		# set missing allele frequencies to zero
		if "gnomAD_v2_exome_AF_popmax" in x_nonMissenseSNV.columns:
			x_nonMissenseSNV["gnomAD_v2_exome_AF_popmax"] = x_nonMissenseSNV["gnomAD_v2_exome_AF_popmax"].fillna(0.0)



		# impute the missing data
		logging.info("Impute the data")
		imp_nonMissenseSNV = SimpleImputer(strategy = 'median')
		imp_nonMissenseSNV.fit(x_nonMissenseSNV)

		imp_nonMissenseSNV.feature_names = list(x_nonMissenseSNV.columns.values)

		x_nonMissenseSNV_imp = imp_nonMissenseSNV.transform(x_nonMissenseSNV)

		with open(args.tempDir + outputPrefix+'.imp_nonMissenseSNV.pkl', 'wb') as f:
			pickle.dump(imp_nonMissenseSNV, f)


		# scale the data
		logging.info("Scaling to [0,1]")
		scal_nonMissenseSNV = MinMaxScaler(clip = 'true')
		scal_nonMissenseSNV.fit(x_nonMissenseSNV_imp)

		scal_nonMissenseSNV.feature_names = list(x_nonMissenseSNV.columns.values)

		x_nonMissenseSNV_imp_scal = scal_nonMissenseSNV.transform(x_nonMissenseSNV_imp)

		with open(args.tempDir + outputPrefix+'.scal_nonMissenseSNV.pkl', 'wb') as f:
			pickle.dump(scal_nonMissenseSNV, f)



		# run logistic regression
		logging.info("Run logistic regression")
		logReg_nonMissenseSNV = LogisticRegression(penalty = 'none')
		logReg_nonMissenseSNV.fit(x_nonMissenseSNV_imp_scal, y_nonMissenseSNV)

		logReg_nonMissenseSNV.feature_names = list(x_nonMissenseSNV.columns.values)

		with open(args.tempDir + outputPrefix+'.logReg_nonMissenseSNV.pkl', 'wb') as f:
			pickle.dump(logReg_nonMissenseSNV, f)

		with open(args.tempDir + outputPrefix+'.predictors_nonMissenseSNV.npy', 'wb') as f:
			np.save(f, x_nonMissenseSNV.columns)

		results_nonMissenseSNV = logReg_nonMissenseSNV.predict_proba(x_nonMissenseSNV_imp_scal)[:,1]

		with open(args.tempDir + outputPrefix + ".nonMissenseSNV_coef.txt", 'a') as f:
		 pprint.pprint(list(zip(logReg_nonMissenseSNV.feature_names, np.round(logReg_nonMissenseSNV.coef_.flatten(), 6))), f)
		 print("Intercept: ", np.round(logReg_nonMissenseSNV.intercept_, 6), file=f)
	else:
		msg = "Not enough non-missense SNVs in training data, using flat prior: " + str(np.round(flatPriors['nonMissenseSNV'], 6))
		logging.info(msg)
		results_nonMissenseSNV = np.full(len(x_nonMissenseSNV.index), flatPriors['nonMissenseSNV'])


	logging.info(" ")



	# gather the results
	pred_prob = np.concatenate( (results_indel, results_missense, results_nonMissenseSNV) )



	prior_prob = pd.DataFrame()
	prior_prob["alleleID"] = np.concatenate((alleleID_indel, alleleID_missense, alleleID_nonMissenseSNV))
	prior_prob["Gene"] = np.concatenate((geneCV_indel, geneCV_missense, geneCV_nonMissenseSNV))
	prior_prob["csqCV"] = np.concatenate((x_indel_csq, x_missense_csq, x_nonMissenseSNV_csq))
	prior_prob["setCV"] = np.concatenate((y_indel, y_missense, y_nonMissenseSNV))
	prior_prob["prior"] = pred_prob.flatten()


	#x_indel_imp_scal_df = pd.DataFrame(x_indel_imp_scal, index = x_indel_index, columns = x_indel.columns)
	#x_missense_imp_scal_df = pd.DataFrame(x_missense_imp_scal, index = x_missense_index, columns = x_missense.columns)
	#x_nonMissenseSNV_imp_scal_df = pd.DataFrame(x_nonMissenseSNV_imp_scal, index = x_nonMissenseSNV_index, columns = x_nonMissenseSNV.columns)

	#x_imp_scal_df = pd.concat([x_indel_imp_scal_df, x_missense_imp_scal_df, x_nonMissenseSNV_imp_scal_df], ignore_index=True, sort=False)

	#combined = pd.concat([x_imp_scal_df, prior_prob], axis=1)
	#combined.to_csv(outputPrefix+".priors.txt", index=False, sep='\t', na_rep='.')




	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")

	






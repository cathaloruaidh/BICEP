import cProfile
import logging
import math
import os
import pickle
import pprint
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from cyvcf2 import VCF, Writer
from joblib import dump, load
from matplotlib.transforms import Affine2D
from pathlib import Path
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor






def generate_prior(x, y, label, args):


	msg = str(np.size(y)) + " " + label + " used (" + str(sum(y == 'PATHOGENIC')) + " PATH, " + str(sum(y == 'BENIGN')) + " BEN)"
	logging.info(msg)

	# set missing allele frequencies to zero
	assert(args.frequency in x.columns), "Allele frequency missing for " + label
	x[args.frequency] = x[args.frequency].fillna(0.0)



	# remove columns that are all NA
	x = x.dropna(axis=1, how='all')


	# evaluate the prior
	if args.eval:
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)


		msg = "Impute the training data (" + label + ")"
		logging.debug(msg)

		imp = SimpleImputer(strategy = 'median', verbose = 100)
		imp.fit(x_train)
		x_train_imp = imp.transform(x_train)
		x_test_imp = imp.transform(x_test)
		with open(args.tempDir + args.prefix + ".TRAIN." + label + '_imp.pkl', 'wb') as f:
			pickle.dump(imp, f)

		msg = "Scaling training data to [0,1] (" + label + ")"
		logging.debug(msg)
		scal = MinMaxScaler(clip = 'true')
		scal.fit(x_train_imp)
		x_train_imp_scal = scal.transform(x_train_imp)
		x_test_imp_scal = scal.transform(x_test_imp)
		with open(args.tempDir + args.prefix + ".TRAIN." + label + '_scal.pkl', 'wb') as f:
			pickle.dump(scal, f)


		msg = "Regression and VIF (" + label + ")"
		logging.debug(msg)
		logReg = LogisticRegression(penalty = 'none')
		logReg.fit(x_train_imp_scal, y_train)
		with open(args.tempDir + args.prefix + ".TRAIN." + label + '_logReg.pkl', 'wb') as f:
			pickle.dump(logReg, f)
		
		with open(args.tempDir + args.prefix + ".TRAIN." + label + '_predictors.npy', 'wb') as f:
			np.save(f, x.columns)


		vif_data = pd.DataFrame()
		vif_data["Model"] =  [ label ] * len(x_train.columns)
		vif_data["Feature"] = x_train.columns
		
		if len(x_train.columns) == 1:
			vif_data["VIF"] = 1
		else:
			vif_data["VIF"] = [variance_inflation_factor(x_train_imp_scal, i) for i in range(len(x_train.columns))]
		vif_data["Coefficient"] = logReg.coef_.flatten()
		vif_data.loc[len(vif_data)] = [ label, "intercept", np.nan, round(logReg.intercept_[0], 4) ]



		SENS_train_boot = []
		SPEC_train_boot = []
		PPV_train_boot = []
		NPV_train_boot = []
		MCC_train_boot = []

		SENS_test_boot = []
		SPEC_test_boot = []
		PPV_test_boot = []
		NPV_test_boot = []
		MCC_test_boot = []


		for i in range(args.boot):
			ind = np.random.randint(x_train_imp_scal.shape[0], size=x_train_imp_scal.shape[0])
			x_boot = x_train_imp_scal[ind]
			y_boot = y_train[ind]

			y_pred = logReg.predict(x_boot)
			tn, fp, fn, tp = confusion_matrix(y_boot, y_pred).ravel()
			SENS_train_boot.append(tp / (tp + fn)) 
			SPEC_train_boot.append(tn / (tn + fp)) 
			PPV_train_boot.append(tp / (tp + fp)) 
			NPV_train_boot.append(tn / (tn + fn)) 
			MCC_train_boot.append(matthews_corrcoef(y_boot, y_pred))


			ind = np.random.randint(x_test_imp_scal.shape[0], size=x_test_imp_scal.shape[0])
			x_boot = x_test_imp_scal[ind]
			y_boot = y_test[ind]

			y_pred = logReg.predict(x_boot)
			tn, fp, fn, tp = confusion_matrix(y_boot, y_pred).ravel()
			SENS_test_boot.append(tp / (tp + fn)) 
			SPEC_test_boot.append(tn / (tn + fp)) 
			PPV_test_boot.append(tp / (tp + fp)) 
			NPV_test_boot.append(tn / (tn + fn)) 
			MCC_test_boot.append(matthews_corrcoef(y_boot, y_pred))


		y_train_pred = logReg.predict(x_train_imp_scal)
		tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()


		performance = pd.DataFrame()

		performance["Model"] = [label]*5
		performance["Data"] = ["Train"]*5
		performance["Metric"] = ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"]
		performance["Value"] = [ tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn), matthews_corrcoef(y_train, y_train_pred) ]
		performance["Value"] = [ round(x, 6) for x in performance["Value"] ]
		performance["L-CI-95"] = [ np.quantile(SENS_train_boot, 0.025), np.quantile(SPEC_train_boot, 0.025), np.quantile(PPV_train_boot, 0.025), np.quantile(NPV_train_boot, 0.025), np.quantile(MCC_train_boot, 0.025)  ]
		performance["L-CI-95"] = [ round(x, 6) for x in performance["L-CI-95"] ]
		performance["U-CI-95"] = [ np.quantile(SENS_train_boot, 0.975), np.quantile(SPEC_train_boot, 0.975), np.quantile(PPV_train_boot, 0.975), np.quantile(NPV_train_boot, 0.975), np.quantile(MCC_train_boot, 0.975)  ]
		performance["U-CI-95"] = [ round(x, 6) for x in performance["U-CI-95"] ]



		y_test_pred = logReg.predict(x_test_imp_scal)
		df = pd.DataFrame(data={'Label' : y_test.flatten(), 'p' : logReg.predict_proba(x_test_imp_scal)[:,1]})
		df.to_csv(args.tempDir + args.prefix + ".TEST." + label + '_priors.txt', index=False, sep="\t")
		tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

		p_tmp = pd.DataFrame()

		p_tmp["Model"] = [label]*5
		p_tmp["Data"] = ["Test"]*5
		p_tmp["Metric"] = ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"]
		p_tmp["Value"] = [ tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn), matthews_corrcoef(y_test, y_test_pred) ]
		p_tmp["Value"] = [ round(x, 6) for x in p_tmp["Value"] ]
		p_tmp["L-CI-95"] = [ np.quantile(SENS_test_boot, 0.025), np.quantile(SPEC_test_boot, 0.025), np.quantile(PPV_test_boot, 0.025), np.quantile(NPV_test_boot, 0.025), np.quantile(MCC_test_boot, 0.025)  ]
		p_tmp["L-CI-95"] = [ round(x, 6) for x in p_tmp["L-CI-95"] ]
		p_tmp["U-CI-95"] = [ np.quantile(SENS_test_boot, 0.975), np.quantile(SPEC_test_boot, 0.975), np.quantile(PPV_test_boot, 0.975), np.quantile(NPV_test_boot, 0.975), np.quantile(MCC_test_boot, 0.975)  ]
		p_tmp["U-CI-95"] = [ round(x, 6) for x in p_tmp["U-CI-95"] ]
		performance = pd.concat([performance, p_tmp], ignore_index = True)

		print(performance)

	# impute the missing data
	logging.info("Impute the data")
	imp = SimpleImputer(strategy = 'median')
	imp.fit(x)

	imp.feature_names = list(x.columns.values)
	x_imp = imp.transform(x)

	with open(args.tempDir + args.prefix + "." + label + '_imp.pkl', 'wb') as f:
		pickle.dump(imp, f)

	
	# scale the data
	logging.info("Scaling to [0,1]")
	scal = MinMaxScaler(clip = 'true')
	scal.fit(x_imp)

	scal.feature_names = list(x.columns.values)
	x_imp_scal = scal.transform(x_imp)

	with open(args.tempDir + args.prefix + "." + label + '_scal.pkl', 'wb') as f:
		pickle.dump(scal, f)



	# run logistic regression
	logging.info("Run logistic regression")
	logReg = LogisticRegression(penalty = 'none')
	logReg.fit(x_imp_scal, y)

	logReg.feature_names = list(x.columns.values)

	with open(args.tempDir + args.prefix + "." + label + '_logReg.pkl', 'wb') as f:
		pickle.dump(logReg, f)

	with open(args.tempDir + args.prefix + "." + label + '_predictors.npy', 'wb') as f:
		np.save(f, x.columns)

	
	with open(args.tempDir + args.prefix + "." + label + "_coef.txt", 'a') as f:
		pprint.pprint(list(zip(logReg.feature_names, np.round(logReg.coef_.flatten(), 6))), f)
		print("Intercept: ", np.round(logReg.intercept_, 6), file=f)


	if args.eval:
		return performance, vif_data







# main function
def PT_main(args):

	logging.info("PRIOR - TRAIN")
	logging.info(" ")


	# ignore warnings
	warnings.filterwarnings("ignore")
	pd.set_option('display.max_columns', None)



	# command line arguments
#	clinVarFullVcfFile = None
	clinVarAnnoVcfFile = None

	if args.clinvar is not None:
		clinVarAnnoVcfFile = args.clinvar

	else:
		if args.cnv is False:
			if args.clinvarPrefix is None:
				clinVarAnnoVcfFile = args.scriptDir + "../data/clinvar_20231126." + args.build + ".PATH_BEN.single.strip.vep.vcf.gz"
			else:
				clinVarAnnoVcfFile = args.scriptDir + "../data/" + args.clinvarPrefix + "." + args.build + ".PATH_BEN.single.strip.vep.vcf.gz"

		else:
			clinVarAnnoVcfFile = args.scriptDir + "../data/nstd102." + args.build + ".variant_call.CNV.PATH_BEN.annotated.vcf.gz"

#	if args.clinvarFull is not None:
#		clinVarFullVcfFile = args.clinvarFull



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

	vepIMPACTRank = {'HIGH' : 1,
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



	# dictionary for flat prior
	flatPriors = {}






	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")





	# parse data from annotated ClinVar file 
	logging.info("Parsing the annotation information")
	CV_anno_vcf = VCF(clinVarAnnoVcfFile, gts012=True)

	if args.cnv:
		keys = re.sub('^.*?: ', '', CV_anno_vcf.get_header_type('CSQ')['Description']).split("|")
		keys = [ key.strip("\"") for key in keys ]

		
		# allele frequency predictor for prior
		if args.frequency is None:
			args.frequency = "gnomAD_v4.1_Max_PopMax_AF"


		if args.predictors is not None:
			d = {}

			with open(args.predictors, 'r') as f:
				for line in f:
					(model, key)=line.split()
					d[model, key] = 1

			# manually add allele frequency to the predictors
			keysPredictors = [ x[1] for x in d.keys()] + [ args.frequency ]
			keysPredictors = sorted(list(set(keysPredictors)))

			keysPredictors_DEL = [ f for m,f in d.keys() if m == "DEL" ] + [args.frequency]
			keysPredictors_DUP = [ f for m,f in d.keys() if m == "DUP" ] + [args.frequency]


		else:
			# TODO update this
			keysPredictors = sorted([ "overlap_loeuf_sumRecip",  args.frequency ])


			keysPredictors_DEL = keysPredictors
			keysPredictors_DUP = keysPredictors


	# SNVs and indels
	else:
		keys = re.sub('^.*?: ', '', CV_anno_vcf.get_header_type('CSQ')['Description']).split("|")
		keys = [ key.strip("\"") for key in keys ]

		
		# allele frequency predictor for prior
		if args.frequency is None:
			#args.frequency = "gnomAD_v2_exome_AF_popmax"
			args.frequency = "gnomADe_AF"


		if args.predictors is not None:
			d = {}

			with open(args.predictors, 'r') as f:
				for line in f:
					(model, key, value)=line.split()
					d[model, key] = value
	
			# manually add allele frequency to the predictors
			keysPredictors = sorted(list(set([ x[1] for x in d.keys()])))
			keysDescPred = sorted(list(set([ f for m,f in d.keys() if d[m,f] == "L" ] + [args.frequency])))
			keysAscPred = sorted(list(set([ f for m,f in d.keys() if d[m,f] == "H" ])))

			keysPredictors = sorted( keysDescPred + keysAscPred )
			keysPredictors = sorted(list(set(keysPredictors)))

			keysPredictors_IND = [ f for m,f in d.keys() if m == "IND" ] + [args.frequency]
			keysPredictors_MIS = [ f for m,f in d.keys() if m == "MIS" ] + [args.frequency]
			keysPredictors_OTH = [ f for m,f in d.keys() if m == "OTH" ] + [args.frequency]


		else:
			#keysPredictors = sorted([ "FATHMM_score", "MPC_score", "Polyphen2_HDIV_score", "REVEL_score", "SIFT_score" ] + [ args.frequency ] )
			#keysDescPred = sorted([ "FATHMM_score", "SIFT_score" ] + [ args.frequency ])
			keysPredictors = sorted([ "fathmm-XF_coding_score", "MPC_score", "PolyPhen2", "REVEL", "SIFT" ] + [ args.frequency ] )
			keysDescPred = sorted([ "fathmm-XF_coding_score", "SIFT" ] + [ args.frequency ])
			keysAscPred  = sorted([ x for x in keysPredictors if x not in keysDescPred ])


			keysPredictors_IND = [ args.frequency ]
			keysPredictors_MIS = sorted([ "fathmm-XF_coding_score", "MPC_score", "PolyPhen", "REVEL", "SIFT" ] + [ args.frequency ])
			keysPredictors_OTH = [ args.frequency ]





	# save variants in list 
	DATA = []
	CHROMS = set([ "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22" ])
	for variant in CV_anno_vcf:
		# remove variants not on autosomes
		if len( set([variant.CHROM, "chr"+variant.CHROM]) & CHROMS ) == 0:
			continue


		if args.cnv:

			# get the ID of the variant
			ID = variant.CHROM + "_" + str(variant.start+1) + "_" + str(variant.INFO.get('END')) + "_" + variant.INFO.get('SVTYPE')
			
			msg = "Reading variant " + ID
			logging.debug(msg)



			# get clinvar allele ID
			alleleID = variant.ID




			# get the clinvar significance, ignoring anything after a comma
			sigCV = variant.INFO.get('CLNSIG').split("\"")[1]



			# get the short significance
			if (sigCV == "Benign") or (sigCV == "Benign%2FLikely_benign") or (sigCV == "Likely%20benign"):
				setCV = "BENIGN"
			elif (sigCV == "Likely%20pathogenic") or (sigCV == "Pathogenic%2FLikely%20pathogenic") or (sigCV == "Pathogenic"):
				setCV = "PATHOGENIC"


			# get the variant type
			typeCV = variant.INFO.get('SVTYPE')


			# get the variant length
			SVLEN = variant.INFO.get('SVLEN')

			# get vep consequences and create a dictionary
			CSQ = variant.INFO.get('CSQ')


			# extract predictors from CSQ and add to DATA
			tmp = np.array(CSQ.split("|"))
			dictVEP = dict(zip(keys, tmp.T))
			l = [ ID, setCV, typeCV, alleleID, SVLEN ]
			for key in keysPredictors:
				l.append(dictVEP[key])
			DATA.append(l) 



		# SNVs and indels
		else:
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


			# remove non-coding variants
			if csqCV not in vepCSQRankCoding.keys():
				continue



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
					if not isinstance(dictVEP[key], float):
						l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
						if len(l) == 0:
							dictVEP[key] = np.nan
						else:
							dictVEP[key] = max(l)

				for key in keysDescPred:
					if not isinstance(dictVEP[key], float):
						l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
						if len(l) == 0:
							dictVEP[key] = np.nan
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
				l = [ ID, setCV, geneCV, csqCV, vepCSQImpact[ csqCV ], typeCV, alleleID ]

		
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
			#	with open(args.prefix + ".err", 'a') as f:
			#		print(ID, "\t", setCV, "\t", csqCV, file=f)

	
	start = 4 if args.cnv else 6

	for i in range(len(DATA)):
		for j in range(start, len(DATA[i])):
			if DATA[i][j] == "." or DATA[i][j] == "":
				DATA[i][j] = np.nan
			else:
				DATA[i][j] = float(DATA[i][j])
			

	# get the IDs of the variants in the pedigree VCF file
	ped_vcf = VCF(args.vcf, gts012=True)

	ped_ID = []
	if args.cnv:
		for variant in ped_vcf:
			ped_ID.append(variant.CHROM + "_" + str(variant.start+1) + "_" + str(variant.INFO.get('END')) + "_" + variant.INFO.get('SVTYPE'))
	else:
		for variant in ped_vcf:
			ped_ID.append(variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0])




	################################################################################
	# Run the Regression model
	################################################################################


	# read in the data
	logging.debug("Converting to DataFrame")
	if args.cnv:
		df = pd.DataFrame(DATA, columns = [ 'ID', 'setCV', 'typeCV', 'alleleID', 'SVLEN' ] + keysPredictors )

	else:
		df = pd.DataFrame(DATA, columns = [ 'ID', 'setCV', 'geneCV', 'csqCV', 'impactCV', 'typeCV', 'alleleID' ] + keysAscPred + keysDescPred )
	
	df['alleleID'] = df.alleleID.astype(str).replace('\.0', '', regex=True)


	# if there is overlap between the ClinVar and pedigree data,
	# remove the variants from the Clinvar data prior to training. 
	df = df[~df['ID'].isin(ped_ID)]



	# if there are pathogenic/benign files, subset to these
	if (args.pathogenic is not None) and (args.benign is not None):
		logging.info("Setting the pathogenic/benign training elements from input file")

		df['setCV'] = "."

		with open(args.pathogenic, 'r') as f:
			pathogenicList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df.loc[df.alleleID.isin(pathogenicList.alleleID), 'setCV'] = "PATHOGENIC"

		with open(args.benign, 'r') as f:
			benignList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df.loc[df.alleleID.isin(benignList.alleleID), 'setCV'] = "BENIGN"

		df = df.drop(df[df['setCV'] == "."].index)



	if args.pathogenic is not None and args.benign is None:
		logging.warning("Pathogenic file is not specified, ignoring benign file")
	
	
	if args.pathogenic is None and args.benign is not None:
		logging.warning("Benign file is not specified, ignoring pathogenic file")




	# if there is an exclude file, remove them 
	if args.exclude is not None:
		logging.info("Removing variants in exclude file")
		with open(args.exclude, 'r') as f:
			excludeList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df = df[~df.alleleID.isin(excludeList.alleleID)]


	# if there is an include file, remove everything else 
	if args.include is not None:
		logging.info("Susbetting to variants in include file")
		with open(args.include, 'r') as f:
			includeList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df = df[df.alleleID.isin(includeList.alleleID)]


	

	
	msg = str(len(df.index)) + " training variants used (" + str((df.setCV.values == 'PATHOGENIC').sum()) + " PATH, " + str((df.setCV.values == 'BENIGN').sum()) + " BEN)"

	logging.info(msg)





	# subset to variables being used
	if args.cnv:
		x = df.filter([ 'ID', 'typeCV', 'alleleID' ] + keysPredictors)
		y = df.filter(['setCV', 'typeCV'])

		logging.info(" ")


		## DELETIONS 
		logging.info("DEL")
		
		x_DEL = x[ x['typeCV'] == 'DEL' ]
		ID_DEL = x_DEL['ID']
		alleleID_DEL = x_DEL['alleleID']
		x_DEL = x_DEL.drop(['ID', 'alleleID', 'typeCV'], axis=1, errors='ignore')

		x_DEL_index = x_DEL.index
		
		

		y_DEL = y[ y['typeCV'] == 'DEL' ]
		y_DEL = y_DEL.drop(['typeCV'], axis=1, errors='ignore')
		y_DEL = y_DEL.values.reshape(-1,1)


		uniq, counts = np.unique(y_DEL, return_counts = True)

		if (len(x_DEL.index) > 0) and (len(uniq) == 2) and (counts.min() > 10*len(x_DEL.columns)):
			drop_cols = [ x for x in keysPredictors if x not in keysPredictors_DEL ]
			x_DEL = x_DEL.drop(drop_cols, axis=1, errors='ignore')

			
			if args.eval:
				perf_DEL, vif_DEL = generate_prior(x_DEL, y_DEL, "DEL", args)

			else:
				generate_prior(x_DEL, y_DEL, "DEL", args)
				


		else:
			if 'DEL' in flatPriors.keys():
				msg = "Not enough DEL in training data, using flat prior: " + str(np.round(flatPriors['DEL'], 6))
				results_DEL = np.full(len(x_DEL.index), flatPriors['DEL'])

			else:
				msg = "Not enough DEL in training data, all DEL will be ignored"
				results_DEL = np.full(len(x_DEL.index), np.nan)

			logging.info(msg)

			if args.eval:
				perf_DEL = pd.DataFrame()

				perf_DEL["Model"] = ["DEL"]*10
				perf_DEL["Data"] = [ y for x in ["Train", "Test"] for y in (x,)*5]
				perf_DEL["Metric"] = [ y for x in ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"] for y in (x,)*2 ]
				perf_DEL["Value"] = [ np.nan ]*10
				perf_DEL["L-CI-95"] = [ np.nan ]*10
				perf_DEL["U-CI-95"] = [ np.nan ]*10





		## DUPLICATIONS
		logging.info("DUP")
		
		x_DUP = x[ x['typeCV'] == 'DUP' ]
		ID_DUP = x_DUP['ID']
		alleleID_DUP = x_DUP['alleleID']
		x_DUP = x_DUP.drop(['ID', 'alleleID', 'typeCV'], axis=1, errors='ignore')

		x_DUP_index = x_DUP.index
		
		

		y_DUP = y[ y['typeCV'] == 'DUP' ]
		y_DUP = y_DUP.drop(['typeCV'], axis=1, errors='ignore')
		y_DUP = y_DUP.values.reshape(-1,1)


		uniq, counts = np.unique(y_DUP, return_counts = True)

		if (len(x_DUP.index) > 0) and (len(uniq) == 2) and (counts.min() > 10*len(x_DUP.columns)):
			drop_cols = [ x for x in keysPredictors if x not in keysPredictors_DUP ]
			x_DUP = x_DUP.drop(drop_cols, axis=1, errors='ignore')

			
			if args.eval:
				perf_DUP, vif_DUP = generate_prior(x_DUP, y_DUP, "DUP", args)

			else:
				generate_prior(x_DUP, y_DUP, "DUP", args)
				


		else:
			if 'DUP' in flatPriors.keys():
				msg = "Not enough DUP in training data, using flat prior: " + str(np.round(flatPriors['DUP'], 6))
				results_DUP = np.full(len(x_DUP.index), flatPriors['DUP'])

			else:
				msg = "Not enough DUP in training data, all DUP will be ignored"
				results_DUP = np.full(len(x_DUP.index), np.nan)

			logging.info(msg)

			if args.eval:
				perf_DUP = pd.DataFrame()

				perf_DUP["Model"] = ["DUP"]*10
				perf_DUP["Data"] = [ y for x in ["Train", "Test"] for y in (x,)*5]
				perf_DUP["Metric"] = [ y for x in ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"] for y in (x,)*2 ]
				perf_DUP["Value"] = [ np.nan ]*10
				perf_DUP["L-CI-95"] = [ np.nan ]*10
				perf_DUP["U-CI-95"] = [ np.nan ]*10




		logging.info(" ")
		logging.info(" ")


	else:
		x = df.filter([ 'ID', 'geneCV', 'csqCV', 'impactCV', 'typeCV', 'alleleID' ] + keysAscPred + keysDescPred)


		y = df.filter(['setCV', 'csqCV', 'typeCV'])


		logging.info(" ")


		## INDELS
		logging.info("IND")
		
		x_IND = x[ x['typeCV'] == 'indel' ]
		ID_IND = x_IND['ID']
		alleleID_IND = x_IND['alleleID']
		geneCV_IND = x_IND['geneCV']
		x_IND = x_IND.drop(['ID', 'alleleID', 'csqCV', 'geneCV', 'typeCV'], axis=1, errors='ignore')

		x_IND_index = x_IND.index
		x_IND['impactCV'] = pd.Categorical(x_IND['impactCV'], categories = sorted(x_IND['impactCV'].unique()))
		
		impactCV_IND = [ x for x in x_IND['impactCV'] if x in vepIMPACTRank.keys() ]
		d_IND = dict((k, vepIMPACTRank[k]) for k in impactCV_IND)
		drop_IND = "impactCV_" + max(d_IND, key=d_IND.get)
		
		x_IND = pd.get_dummies(x_IND, columns = ['impactCV']).drop(drop_IND, axis=1)


		y_IND = y[ y['typeCV'] == 'indel' ]
		y_IND = y_IND.drop(['csqCV', 'impactCV', 'typeCV'], axis=1, errors='ignore')
		y_IND = y_IND.values.reshape(-1,1)


		uniq, counts = np.unique(y_IND, return_counts = True)

		if (len(x_IND.index) > 0) and (len(uniq) == 2) and (counts.min() > 10*len(x_IND.columns)):
			drop_cols = [ x for x in keysPredictors if x not in keysPredictors_IND ]
			x_IND = x_IND.drop(drop_cols, axis=1, errors='ignore')
			#x_IND = x_IND[x_IND.columns.intersection(keysPredictors_IND)]

			
			if args.eval:
				perf_IND, vif_IND = generate_prior(x_IND, y_IND, "IND", args)

			else:
				generate_prior(x_IND, y_IND, "IND", args)
				


		else:
			if 'IND' in flatPriors.keys():
				msg = "Not enough IND in training data, using flat prior: " + str(np.round(flatPriors['IND'], 6))
				results_IND = np.full(len(x_IND.index), flatPriors['IND'])

			else:
				msg = "Not enough IND in training data, all IND will be ignored"
				results_IND = np.full(len(x_IND.index), np.nan)

			logging.info(msg)

			if args.eval:
				perf_IND = pd.DataFrame()

				perf_IND["Model"] = ["IND"]*10
				perf_IND["Data"] = [ y for x in ["Train", "Test"] for y in (x,)*5]
				perf_IND["Metric"] = ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"]*2
				perf_IND["Value"] = [ np.nan ]*10
				perf_IND["L-CI-95"] = [ np.nan ]*10
				perf_IND["U-CI-95"] = [ np.nan ]*10
				print(perf_IND)

				vif_IND = pd.DataFrame()
				vif_IND["Model"] = "IND"
				vif_IND["Feature"] = "NA"
				vif_IND["VIF"] = np.nan
				vif_IND["Coefficient"] = np.nan





		logging.info(" ")
		logging.info(" ")



		## MISSENSE SNV
		logging.info("MIS")
		x_MIS = x[ (x['csqCV'] == 'missense_variant') & (x['typeCV'] == 'SNV') ]
		x_MIS_csq = x_MIS['csqCV'] 
		x_MIS = x_MIS.drop(['typeCV', 'csqCV', 'impactCV'], axis=1, errors='ignore')



		y_MIS = y[ (y['csqCV'] == 'missense_variant') & (y['typeCV'] == 'SNV') ]
		y_MIS = y_MIS.drop(['csqCV', 'typeCV'], axis=1)
		y_MIS = y_MIS.values.reshape(-1,1)


		# get IDs
		ID_MIS = x_MIS['ID']
		alleleID_MIS = x_MIS['alleleID']
		geneCV_MIS = x_MIS['geneCV']
		x_MIS = x_MIS.drop(['ID', 'alleleID', 'geneCV'], axis=1)
		x_MIS_index = x_MIS.index


		uniq, counts = np.unique(y_MIS, return_counts = True)

		if (len(x_MIS.index) > 0) and (len(uniq) == 2) and (counts.min() > 15*len(x_MIS.columns)):
			drop_cols = [ x for x in keysPredictors if x not in keysPredictors_MIS ]
			x_MIS = x_MIS.drop(drop_cols, axis=1, errors='ignore')

			if args.eval:
				perf_MIS, vif_MIS = generate_prior(x_MIS, y_MIS, "MIS", args)

			else:
				generate_prior(x_MIS, y_MIS, "MIS", args)

		
		else:
			if 'MIS' in flatPriors.keys():
				msg = "Not enough MIS variants in training data, using flat prior: " + str(np.round(flatPriors['MIS'], 6))
				results_MIS = np.full(len(x_MIS.index), flatPriors['MIS'])

			else:
				msg = "Not enough MIS variants in training data, all MIS variants will be ignored"
				results_MIS = np.full(len(x_MIS.index), np.nan)

			logging.info(msg)

			if args.eval:
				perf_MIS = pd.DataFrame()

				perf_MIS["Model"] = ["MIS"]*10
				perf_MIS["Data"] = [ y for x in ["Train", "Test"] for y in (x,)*5 ]
				perf_MIS["Metric"] = [ y for x in ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"] for y in (x,)*2 ]
				perf_MIS["Value"] = [ np.nan ]*10
				perf_MIS["L-CI-95"] = [ np.nan ]*10
				perf_MIS["U-CI-95"] = [ np.nan ]*10

				vif_MIS = pd.DataFrame()
				vif_MIS["Model"] = "MIS"
				vif_MIS["Feature"] = "NA"
				vif_MIS["VIF"] = np.nan
				vif_MIS["Coefficient"] = np.nan





		logging.info(" ")
		logging.info(" ")



		## NON-MISSENSE SNV
		logging.info("NON-MISSENSE SNV")
		x_OTH = x[ (x['csqCV'] != 'missense_variant') & (x['typeCV'] == 'SNV') ]
		x_OTH_csq = x_OTH['csqCV'] 


		y_OTH = y[ (y['csqCV'] != 'missense_variant') & (y['typeCV'] == 'SNV') ]
		y_OTH = y_OTH.drop(['csqCV', 'typeCV'], axis=1)
		y_OTH = y_OTH.values.reshape(-1,1)


		# get IDs
		ID_OTH = x_OTH['ID']
		alleleID_OTH = x_OTH['alleleID']
		geneCV_OTH = x_OTH['geneCV']
		x_OTH = x_OTH.drop(['ID', 'alleleID', 'geneCV', 'impactCV', 'typeCV'], axis=1)
		x_OTH_index = x_OTH.index

		x_OTH['csqCV'] = pd.Categorical(x_OTH['csqCV'], categories = sorted(x_OTH['csqCV'].unique()))
		csqCV_OTH = [ x for x in x_OTH['csqCV'] if x in vepCSQRank.keys() ]
		d_OTH = dict((k, vepCSQRank[k]) for k in csqCV_OTH)


		drop_OTH = "csqCV_" + max(d_OTH, key=d_OTH.get)
		x_OTH = pd.get_dummies(x_OTH, columns = ['csqCV']).drop(drop_OTH, axis=1)


		uniq, counts = np.unique(y_OTH, return_counts = True)

		if (len(x_OTH.index) > 0) and (len(uniq) == 2) and (counts.min() > 15*len(x_OTH.columns)):
			drop_cols = [ x for x in keysPredictors if x not in keysPredictors_OTH ]
			x_OTH = x_OTH.drop(drop_cols, axis=1, errors='ignore')


			if args.eval:
				perf_OTH, vif_OTH = generate_prior(x_OTH, y_OTH, "OTH", args)

			else:
				generate_prior(x_OTH, y_OTH, "OTH", args)



		else:
			if 'OTH' in flatPriors.keys():
				msg = "Not enough OTH in training data, using flat prior: " + str(np.round(flatPriors['OTH'], 6))
				results_OTH = np.full(len(x_OTH.index), flatPriors['OTH'])

			else:
				msg = "Not enough OTH in training data, all OTH will be ignored"
				results_OTH = np.full(len(x_OTH.index), np.nan)

			logging.info(msg)

			if args.eval:
				perf_OTH = pd.DataFrame()

				perf_OTH["Model"] = ["OTH"]*10
				perf_OTH["Data"] =  [ y for x in ["Train", "Test"] for y in (x,)*5] 
				perf_OTH["Metric"] = [ y for x in ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"] for y in (x,)*2 ]
				perf_OTH["Value"] = [ np.nan ]*10
				perf_OTH["L-CI-95"] = [ np.nan ]*10
				perf_OTH["U-CI-95"] = [ np.nan ]*10

				vif_OTH = pd.DataFrame()
				vif_OTH["Model"] = "OTH"
				vif_OTH["Feature"] = "NA"
				vif_OTH["VIF"] = np.nan
				vif_OTH["Coefficient"] = np.nan




	#  plot the evaluation metrics for the prior 
	if args.eval:
		if args.cnv:
			x = ["DEL", "DUP"]
			val_tmp = [1, 1]
			performance = pd.concat([ perf_DEL, perf_DUP ], ignore_index = True)


		else:
			x = ["IND", "MIS", "OTH"]
			val_tmp = [1, 1, 1]
			performance = pd.concat([ perf_IND, perf_MIS, perf_OTH ], ignore_index = True)

		performance["L_Err"] = performance["Value"] - performance["L-CI-95"]
		performance["U_Err"] = performance["U-CI-95"] - performance["Value"]
		performance.to_csv(args.tempDir + args.prefix + ".performance.txt", index=False, sep='\t', na_rep='.')


		title_list = [ "Sensitivity", "Specificity", "PPV", "NPV", "MCC" ]

		fig, axs = plt.subplots(1, 5, sharey = True, figsize=(10,3))

		for i in range(5):
			p_train_null = performance[ (performance["Metric"] == title_list[i]) & (performance["Data"] == "Train") & (performance["Value"].isnull()) ]
			p_train_ok = performance[ (performance["Metric"] == title_list[i]) & (performance["Data"] == "Train") & (performance["Value"].notnull()) ]
			p_test_null = performance[ (performance["Metric"] == title_list[i]) & (performance["Data"] == "Test") & (performance["Value"].isnull()) ]
			p_test_ok = performance[ (performance["Metric"] == title_list[i]) & (performance["Data"] == "Test") & (performance["Value"].notnull()) ]

			trans1 = Affine2D().translate(-0.15, 0.0) + axs[i].transData
			trans2 = Affine2D().translate(+0.15, 0.0) + axs[i].transData

			axs[i].errorbar(x, val_tmp, color="white", label="_tmp_")
			axs[i].errorbar(p_train_ok["Model"], p_train_ok["Value"], yerr=[ p_train_ok["L_Err"].values.tolist(), p_train_ok["U_Err"].values.tolist()  ], marker="o", linestyle="none", transform=trans1, capsize=4, label="Train")
			axs[i].errorbar(p_test_ok["Model"], p_test_ok["Value"], yerr=[ p_test_ok["L_Err"].values.tolist(), p_test_ok["U_Err"].values.tolist()  ], marker="^", linestyle="none", transform=trans2, capsize=4, label="Test")
			axs[i].errorbar(p_train_null["Model"], [1]*p_train_null["Value"].isnull().sum(), yerr=0, marker="x", linestyle="none", transform=trans1, color="#7f7f7f", label="NA")
			axs[i].errorbar(p_test_null["Model"], [1]*p_test_null["Value"].isnull().sum(), yerr=0, marker="x", linestyle="none", transform=trans2, color="#7f7f7f", label="NA")
			axs[i].set_title(title_list[i])



		handles, labels = axs[0].get_legend_handles_labels()
		fig.legend(handles[0:2], labels[0:2])
		plt.savefig(args.outputDir + args.prefix + ".metrics_prior.png", dpi=300)

		
		if args.cnv:
			VIF = pd.concat([ vif_DEL, vif_DUP ], ignore_index = True)

		else:
			VIF = pd.concat([ vif_IND, vif_MIS, vif_OTH ], ignore_index = True)

		VIF = VIF.round(3)
		VIF.to_csv(args.tempDir + args.prefix + ".features.txt", index=False, sep='\t', na_rep='.')



	logging.info(" ")






	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")

	






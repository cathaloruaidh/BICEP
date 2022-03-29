#!/usr/bin/python




import cProfile, getopt, logging, math, os, pickle, pprint, re, sys
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, r2_score
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

from joblib import dump, load

from cyvcf2 import VCF, Writer

import warnings





# main function
def main(argv):


	# ignore warnings
	warnings.filterwarnings("ignore")



	# command line arguments
	nCores = 1
	excludeFile = None
	inputVcfFile = None
	outputPrefix = None
	nBoot = 1
	logLevel = None

	try:
		opts, args = getopt.getopt(argv, "b:e:l:o:", ["boot=", "exclude=", "log=", "output=", "vcf="])
	except getopt.GetoptError:
		print("Getopt Error")
		logging.error("getopt error")
		sys.exit("Exiting ... ")

	for opt, arg in opts:
		if opt in ("-b", "--boot"):
			nBoot = int(arg)

		if opt in ("-e", "--exclude"):
			excludeFile = arg
	
		if opt in ("-l", "--log"):
			logLevel = arg.upper()
			numericLevel = getattr(logging, arg.upper(), None)
			if not isinstance(numericLevel, int):
				raise ValueError('Invalid log level: %s' % arg)

		if opt in ("-o", "--output"):
			outputPrefix = arg
	
		if opt in ("-v", "--vcf"):
			inputVcfFile = arg
	


	FORMAT = '# %(asctime)s [%(levelname)s] - %(message)s'
	
	if logLevel is None:
		logging.basicConfig(format=FORMAT)
	else:
		numericLevel = getattr(logging, logLevel, None)
		if not isinstance(numericLevel, int):
			raise ValueError('Invalid log level: %s' % logLevel)
		logging.basicConfig(format=FORMAT, level=logLevel)
	


	if outputPrefix is None:
		outputPrefix = "output"


	with open(outputPrefix + ".log", 'w') as f:
		True



	# add colours to the log name
	logging.addLevelName(logging.NOTSET, "NOT  ")
	logging.addLevelName(logging.DEBUG, "\u001b[36mDEBUG\u001b[0m")
	logging.addLevelName(logging.INFO, "INFO ")
	logging.addLevelName(logging.WARNING, "\u001b[33mWARN \u001b[0m")
	logging.addLevelName(logging.ERROR, "\u001b[31mERROR\u001b[0m")
	logging.addLevelName(logging.CRITICAL, "\u001b[35mCRIT \u001b[0m")




	# define SCHEMA functinoal consequences 
	vepCSQRank = {'splice_acceptor_variant' : 1, 
	'splice_donor_variant' : 2, 
	'stop_gained' : 3, 
	'frameshift_variant' : 4, 
	'missense_variant' : 5 }








	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")


	# get VCF file and variant/sample information
	vcf = VCF(inputVcfFile, gts012=True)


	keys = re.sub('^.*?: ', '', vcf.get_header_type('CSQ')['Description']).split("|")
	keys = [ key.strip("\"") for key in keys ]
	keysPredictors = [ "MPC_score", "gnomAD_v2_exome_AF_popmax" ]
	



	# save variants in list 
	DATA = []
	for variant in vcf:
		# get the ID of the variant
		ID = variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0] 
		
		msg = "Reading variant " + ID
		logging.debug(msg)



		# get clinvar allele ID
		alleleID = variant.ID




		# get the clinvar significance, ignoring anthing after a comma
		sigCV = variant.INFO.get('CLNSIG').split(",", 1)[0]



		# get the short significance
		if (sigCV == "Benign") or (sigCV == "Benign/Likely_benign") or (sigCV == "Likely_benign"):
			setCV = "BENIGN"
		elif (sigCV == "Likely_pathogenic") or (sigCV == "Pathogenic/Likely_pathogenic") or (sigCV == "Pathogenic"):
			setCV = "PATHOGENIC"


		# get the variant type, ignore if not SNV or indel
		if len(variant.REF) == 1 and len(variant.ALT[0]) == 1:
			typeCV = "SNV"
		elif ( (len(variant.REF) == 1 and len(variant.ALT[0]) > 1) or ((len(variant.REF) > 1 and len(variant.ALT[0]) == 1)) ) and max(len(variant.REF), len(variant.ALT[0])) < 50:
			typeCV = "indel"
		else:
			with open(outputPrefix + ".err", 'a') as f:
				print(ID, "\t", setCV, "\tNA\tTYPE", file=f)
			continue



		# get the clinvar gene list
		tmp = variant.INFO.get('GENEINFO')
		if (tmp is None) or ("|" in tmp):
			with open(outputPrefix + ".err", 'a') as f:
				print(ID, "\t", setCV, "\tNA\tGENEINFO=", tmp, file=f)
			continue
		geneCV = re.sub(':.*?$', '', tmp)



		# get the variant clinvar consequence
		tmp = variant.INFO.get('MC')	
		if (tmp is None) or ("," in tmp):
			with open(outputPrefix + ".err", 'a') as f:
				print(ID, "\t", setCV, "\tNA\tMC=", tmp, file=f)
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
			for key in keysPredictors:
				l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
				if len(l) == 0:
					dictVEP[key] = "."
				else:
					dictVEP[key] = max(l)



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
			l = [ ID, setCV, csqCV, typeCV, alleleID ]

	
			for key in keysPredictors:
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

			DATA.append(l) 


		# otherwise print to error file
		else:
			with open(outputPrefix + ".err", 'a') as f:
				print(ID, "\t", setCV, "\t", csqCV, file=f)

	
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
	df_full = pd.DataFrame(DATA, columns = [ 'ID', 'setCV', 'csqCV', 'typeCV', 'alleleID' ] + keysPredictors )

	df_full.to_csv("DATA.txt", index=False, sep='\t')


	# if there is an exclude file, remove them 
	if excludeFile is None:
		df = df_full
	else:
		logging.info("Removing variants in exclude file")
		with open(excludeFile, 'r') as f:
			excludeList = pd.read_csv(f, header=None, names = ['alleleID'], dtype=str)
			df = df_full[~df_full.alleleID.isin(excludeList.alleleID)]


	
	msg = "A total of " + str(len(df.index)) + " variants used (" + str((df.setCV.values == 'PATHOGENIC').sum()) + " PATH and " + str((df.setCV.values == 'BENIGN').sum()) + " BEN)"

	logging.info(msg)


	### Model 1
	logging.info("Model 1")
	logging.info("All variants: CADD + AF + CSQ + TYPE")

	x = df.filter(['ID', 'alleleID', 'CADD_PHRED', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x_csq = x['csqCV']
	x = pd.get_dummies(x, drop_first = True, columns = ['csqCV', 'typeCV'])


	y = df['setCV']
	y = y.values.reshape(-1,1)




	# get ID
	ID = x['ID']
	alleleID = x['alleleID']
	x_old = x
	x = x.drop(['ID', 'alleleID'], axis=1)


	# apply median-based imputation to the data
	logging.info("Median-based imputation")
	imp = SimpleImputer(strategy = 'median')
	imp.fit(x)

	x_imp = imp.transform(x)

	with open(outputPrefix+'.Model1.imp.pkl', 'wb') as f:
		pickle.dump(imp, f)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal = MinMaxScaler(clip = 'true')
	scal.fit(x_imp)

	x_imp_scal = scal.transform(x_imp)

	with open(outputPrefix+'.Model1.scal.pkl', 'wb') as f:
		pickle.dump(scal, f)



	# run logistic regression
	logging.info("Run the logistic regession")
	logReg = LogisticRegression(penalty = 'none')
	logReg.fit(x_imp_scal, y)

	logReg.feature_names = list(x.columns.values)

	with open(outputPrefix+'.Model1.logReg.pkl', 'wb') as f:
		pickle.dump(logReg, f)

	pprint.pprint(list(zip(logReg.feature_names, logReg.coef_.flatten())))



	# boostrap the metrics
	SENS_boot = []
	SPEC_boot = []
	MCC_boot = []


	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		ind = np.random.randint(x_imp_scal.shape[0], size=x_imp_scal.shape[0])
		x_boot = x_imp_scal[ind]
		y_boot = y[ind]

		y_pred = logReg.predict(x_boot)
		tn, fp, fn, tp = confusion_matrix(y_boot, y_pred).ravel()
		SENS_boot.append(tp / (tp + fn)) 
		SPEC_boot.append(tn / (tn + fp)) 
		MCC_boot.append(matthews_corrcoef(y_boot, y_pred))




	# apply to data and get statistics
	y_pred = logReg.predict(x_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

	with open(outputPrefix+".log", 'a') as f:
		print("Model 1", file=f)
		print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_boot, 0.025), ", ", np.quantile(SENS_boot, 0.975), ")", file=f)
		print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_boot, 0.025), ", ", np.quantile(SPEC_boot, 0.975), ")", file=f)
		print("\tMCC - ", matthews_corrcoef(y, y_pred), " (", np.quantile(MCC_boot, 0.025), ", ", np.quantile(MCC_boot, 0.975), ")", file=f)
		print("\n", file=f)



	prior_prob = pd.DataFrame()
	prior_prob["ID"] = ID
	prior_prob["csqCV"] = x_csq
	prior_prob["alleleID"] = alleleID
	prior_prob["prior"] = logReg.predict_proba(x_imp_scal)[:,1]

	x_imp_scal_df = pd.DataFrame(x_imp_scal, index = prior_prob.index, columns = x.columns)

	combined = pd.concat([x_imp_scal_df, prior_prob], axis=1)
	combined.to_csv(outputPrefix+".Model1.priors.txt", index=False, sep='\t')


	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")










	### Model 3 
	logging.info("Model 3")
	logging.info("Missense (AF + MPC) and non-missense (AF + CSQ + TYPE)")


	x = df.filter(['ID', 'alleleID', 'MPC_score', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])


	y = df.filter(['setCV', 'csqCV'])




	# split into missense and nonMissense
	x_missense = x[ x['csqCV'] == 'missense_variant' ]
	x_missense_csq = x_missense['csqCV'] 
	x_missense = x_missense.drop(['typeCV', 'csqCV'], axis=1)


	x_nonMissense = x[ x['csqCV'] != 'missense_variant' ]
	x_nonMissense_csq = x_nonMissense['csqCV'] 
	x_nonMissense['csqCV'] = pd.Categorical(x_nonMissense['csqCV'], categories = sorted(x_nonMissense['csqCV'].unique()))
	x_nonMissense['typeCV'] = pd.Categorical(x_nonMissense['typeCV'], categories = sorted(x_nonMissense['typeCV'].unique()))
	x_nonMissense = x_nonMissense.drop(['MPC_score'], axis=1)
	x_nonMissense = pd.get_dummies(x_nonMissense, drop_first = True, columns = ['csqCV', 'typeCV'])



	y_missense = y[ y['csqCV'] == 'missense_variant' ]
	y_missense = y_missense.drop(['csqCV'], axis=1)
	y_missense = y_missense.values.reshape(-1,1)


	y_nonMissense = y[ y['csqCV'] != 'missense_variant' ]
	y_nonMissense = y_nonMissense.drop(['csqCV'], axis=1)
	y_nonMissense = y_nonMissense.values.reshape(-1,1)





	# get IDs
	ID_missense = x_missense['ID']
	alleleID_missense = x_missense['alleleID']
	x_missense = x_missense.drop(['ID', 'alleleID'], axis=1)
	x_missense_index = x_missense.index

	ID_nonMissense = x_nonMissense['ID']
	alleleID_nonMissense = x_nonMissense['alleleID']
	x_nonMissense = x_nonMissense.drop(['ID', 'alleleID'], axis=1)
	x_nonMissense_index = x_nonMissense.index



	# impute the missing data
	logging.info("Impute the data")

	imp_missense = SimpleImputer(strategy = 'median')
	imp_missense.fit(x_missense)
	x_missense_imp = imp_missense.transform(x_missense)

	with open(outputPrefix+'.Model3.imp_missense.pkl', 'wb') as f:
		pickle.dump(imp_missense, f)


	imp_nonMissense = SimpleImputer(strategy = 'median')
	imp_nonMissense.fit(x_nonMissense)
	x_nonMissense_imp = imp_nonMissense.transform(x_nonMissense)

	with open(outputPrefix+'.Model3.imp_nonMissense.pkl', 'wb') as f:
		pickle.dump(imp_nonMissense, f)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal_missense = MinMaxScaler(clip = 'true')
	scal_missense.fit(x_missense_imp)
	x_missense_imp_scal = scal_missense.transform(x_missense_imp)

	with open(outputPrefix+'.Model3.scal_missense.pkl', 'wb') as f:
		pickle.dump(scal_missense, f)


	scal_nonMissense = MinMaxScaler(clip = 'true')
	scal_nonMissense.fit(x_nonMissense_imp)
	x_nonMissense_imp_scal = scal_nonMissense.transform(x_nonMissense_imp)

	with open(outputPrefix+'.Model3.scal_nonMissense.pkl', 'wb') as f:
		pickle.dump(scal_nonMissense, f)




	# run logistic regression
	logging.info("Run logistic regression")

	logReg_missense = LogisticRegression(penalty = 'none')
	logReg_missense.fit(x_missense_imp_scal, y_missense)

	logReg_missense.feature_names = list(x_missense.columns.values)

	with open(outputPrefix+'.Model3.logReg_missense.pkl', 'wb') as f:
		pickle.dump(logReg_missense, f)


	logReg_nonMissense = LogisticRegression(penalty = 'none')
	logReg_nonMissense.fit(x_nonMissense_imp_scal, y_nonMissense)

	logReg_nonMissense.feature_names = list(x_nonMissense.columns.values)

	with open(outputPrefix+'.Model3.logReg_nonMissense.pkl', 'wb') as f:
		pickle.dump(logReg_nonMissense, f)



	pprint.pprint(list(zip(logReg_missense.feature_names, logReg_missense.coef_.flatten())))
	print("\n")
	pprint.pprint(list(zip(logReg_nonMissense.feature_names, logReg_nonMissense.coef_.flatten())))



	SENS_missense_boot = []
	SPEC_missense_boot = []
	MCC_missense_boot = []

	SENS_nonMissense_boot = []
	SPEC_nonMissense_boot = []
	MCC_nonMissense_boot = []

	SENS_COMBINED_boot = []
	SPEC_COMBINED_boot = []
	MCC_COMBINED_boot = []



	# boostrap the metrics
	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		# missense
		ind_missense = np.random.randint(x_missense_imp_scal.shape[0], size=x_missense_imp_scal.shape[0])
		x_missense_boot = x_missense_imp_scal[ind_missense]
		y_missense_boot = y_missense[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_missense_boot.append(tp / (tp + fn)) 
		SPEC_missense_boot.append(tn / (tn + fp)) 
		MCC_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissense
		ind_nonMissense = np.random.randint(x_nonMissense_imp_scal.shape[0], size=x_nonMissense_imp_scal.shape[0])
		x_nonMissense_boot = x_nonMissense_imp_scal[ind_nonMissense]
		y_nonMissense_boot = y_nonMissense[ind_nonMissense]

		y_nonMissense_pred = logReg_nonMissense.predict(x_nonMissense_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissense_boot, y_nonMissense_pred).ravel()
		SENS_nonMissense_boot.append(tp / (tp + fn)) 
		SPEC_nonMissense_boot.append(tn / (tn + fp)) 
		MCC_nonMissense_boot.append(matthews_corrcoef(y_nonMissense_boot, y_nonMissense_pred))


		# COMBINED = missense + nonMissense
		y_COMBINED_pred = np.append(y_missense_pred, y_nonMissense_pred)
		y_COMBINED_boot = np.append(y_missense_boot, y_nonMissense_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_COMBINED_boot.append(tn / (tn + fp)) 
		MCC_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))




	# apply to data and get statistics
	# missense
	logging.info("Results: missense")
	y_missense_pred = logReg_missense.predict(x_missense_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_missense, y_missense_pred).ravel()
	
	with open(outputPrefix+".log", 'a') as f:
		print("Model 3 - missense", file=f)
		print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_missense_boot, 0.025), ", ", np.quantile(SENS_missense_boot, 0.975), ")", file=f)
		print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_missense_boot, 0.025), ", ", np.quantile(SPEC_missense_boot, 0.975), ")", file=f)
		print("\tMCC - ", matthews_corrcoef(y_missense, y_missense_pred), " (", np.quantile(MCC_missense_boot, 0.025), ", ", np.quantile(MCC_missense_boot, 0.975), ")", file=f)
		print("\n", file=f)
	
	logging.info(" ")



	# nonMissense
	logging.info("Results: nonMissense")
	y_nonMissense_pred = logReg_nonMissense.predict(x_nonMissense_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_nonMissense, y_nonMissense_pred).ravel()
	
	with open(outputPrefix+".log", 'a') as f:
		print("Model 3 - nonMissense", file=f)
		print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_nonMissense_boot, 0.025), ", ", np.quantile(SENS_nonMissense_boot, 0.975), ")", file=f)
		print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_nonMissense_boot, 0.025), ", ", np.quantile(SPEC_nonMissense_boot, 0.975), ")", file=f)
		print("\tMCC - ", matthews_corrcoef(y_nonMissense, y_nonMissense_pred), " (", np.quantile(MCC_nonMissense_boot, 0.025), ", ", np.quantile(MCC_nonMissense_boot, 0.975), ")", file=f)
		print("\n", file=f)

	logging.info(" ")



	# COMBINED = missense + nonMissense
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_missense, y_nonMissense)
	y_COMBINED_pred = np.append(y_missense_pred, y_nonMissense_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	with open(outputPrefix+".log", 'a') as f:
		print("Model 3 - COMBINED", file=f)
		print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_COMBINED_boot, 0.025), ", ", np.quantile(SENS_COMBINED_boot, 0.975), ")", file=f)
		print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_COMBINED_boot, 0.975), ")", file=f)
		print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_COMBINED_boot, 0.025), ", ", np.quantile(MCC_COMBINED_boot, 0.975), ")", file=f)
		print("\n", file=f)


	pred_prob = np.append( logReg_missense.predict_proba(x_missense_imp_scal)[:,1], logReg_nonMissense.predict_proba(x_nonMissense_imp_scal)[:,1])



	prior_prob = pd.DataFrame()
	prior_prob["alleleID"] = np.append(alleleID_missense, alleleID_nonMissense).flatten()
	prior_prob["csqCV"] = np.append(x_missense_csq, x_nonMissense_csq).flatten()
	prior_prob["setCV"] = y_COMBINED.flatten()
	prior_prob["prior"] = pred_prob.flatten()

	x_missense_imp_scal_df = pd.DataFrame(x_missense_imp_scal, index = x_missense_index, columns = x_missense.columns)
	x_nonMissense_imp_scal_df = pd.DataFrame(x_nonMissense_imp_scal, index = x_nonMissense_index, columns = x_nonMissense.columns)

	x_imp_scal_df = pd.concat([x_missense_imp_scal_df, x_nonMissense_imp_scal_df], ignore_index=True, sort=False)

	combined = pd.concat([x_imp_scal_df, prior_prob], axis=1)
	combined.to_csv(outputPrefix+".Model3.priors.txt", index=False, sep='\t')




	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")




	logging.info("Done")

	




if __name__ == "__main__":
	main(sys.argv[1:])



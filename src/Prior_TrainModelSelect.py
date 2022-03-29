#!/usr/bin/python




import cProfile, getopt, logging, math, os, pprint, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


import warnings





# main function
def main(argv):


	# ignore warnings
	warnings.filterwarnings("ignore")



	# command line arguments
	nCores = 1
	inputFile = None
	outputFile = None
	outputLog = None
	nBoot = 1

	try:
		opts, args = getopt.getopt(argv, "b:f:l:o:v:", ["boot=", "file=", "log=", "output=", "vcf="])
	except getopt.GetoptError:
		print("Getopt Error")
		logging.error("getopt error")
		sys.exit("Exiting ... ")

	for opt, arg in opts:
		if opt in ("-b", "--boot"):
			nBoot = int(arg)

		if opt in ("-f", "--file"):
			inputFile = arg
	
		if opt in ("-l", "--log"):
			logLevel = arg.upper()
			numericLevel = getattr(logging, arg.upper(), None)
			if not isinstance(numericLevel, int):
				raise ValueError('Invalid log level: %s' % arg)

		if opt in ("-o", "--output"):
			outputFile = arg
	
	FORMAT = '# %(asctime)s [%(levelname)s] - %(message)s'
	
	try:
		logLevel
	except:
		logging.basicConfig(format=FORMAT)
	else:
		numericLevel = getattr(logging, logLevel, None)
		if not isinstance(numericLevel, int):
			raise ValueError('Invalid log level: %s' % logLevel)
		logging.basicConfig(format=FORMAT, level=logLevel)
	

	if outputFile is None:
		outputFile = "output"


	# add colours to the log name
	logging.addLevelName(logging.NOTSET, "NOT  ")
	logging.addLevelName(logging.DEBUG, "\u001b[36mDEBUG\u001b[0m")
	logging.addLevelName(logging.INFO, "INFO ")
	logging.addLevelName(logging.WARNING, "\u001b[33mWARN \u001b[0m")
	logging.addLevelName(logging.ERROR, "\u001b[31mERROR\u001b[0m")
	logging.addLevelName(logging.CRITICAL, "\u001b[35mCRIT\u001b[0m")



	################################################################################
	# Run the Regression model
	################################################################################


	# read in the data
	#df = pd.read_csv("results.Priors.relax.txt", sep="\t", na_values = ['.'])
	df = pd.read_csv(inputFile, sep="\t", na_values = ['.'])


	

	# set missing allele frequencies to zero
	#df["gnomAD_v2_exome_AF_popmax"] = df["gnomAD_v2_exome_AF_popmax"].fillna(0.0)



	### Model 1
	logging.info("Model 1")
	logging.info("All variants: CADD + AF + CSQ + TYPE")

	x = df.filter(['CADD_PHRED', 'GERP', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x = df.filter(['CADD_PHRED', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x = pd.get_dummies(x, drop_first = True, columns = ['csqCV', 'typeCV'])


	y = df['setCV']
	y = y.values.reshape(-1,1)


	# split into training and testing set
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)


	logging.info("Impute the data and run logistic regression")



	# apply median-based imputation to the data
	logging.info("Median-based imputation")
	imp = SimpleImputer(strategy = 'median')
	imp.fit(x_train)

	x_train_imp = imp.transform(x_train)
	x_test_imp = imp.transform(x_test)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal = MinMaxScaler(clip = 'true')
	scal.fit(x_train_imp)

	x_train_imp_scal = scal.transform(x_train_imp)
	x_test_imp_scal = scal.transform(x_test_imp)



	# run logistic regression
	logging.info("Run the logistic regession")
	logReg = LogisticRegression(penalty = 'none')
	logReg.fit(x_train_imp_scal, y_train)

	vif_data = pd.DataFrame()
	vif_data["feature"] = x_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_train_imp_scal, i) for i in range(len(x_train.columns))]
	vif_data["Coefficient"] = logReg.coef_.flatten()
	print(vif_data)
	

	# boostrap the metrics
	SENS_train_boot = []
	SPEC_train_boot = []
	PPV_train_boot = []
	MCC_train_boot = []


	SENS_test_boot = []
	SPEC_test_boot = []
	PPV_test_boot = []
	MCC_test_boot = []


	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		ind = np.random.randint(x_train_imp_scal.shape[0], size=x_train_imp_scal.shape[0])
		x_boot = x_train_imp_scal[ind]
		y_boot = y_train[ind]

		y_pred = logReg.predict(x_boot)
		tn, fp, fn, tp = confusion_matrix(y_boot, y_pred).ravel()
		SENS_train_boot.append(tp / (tp + fn)) 
		SPEC_train_boot.append(tn / (tn + fp)) 
		PPV_train_boot.append(tp / (tp + fp)) 
		MCC_train_boot.append(matthews_corrcoef(y_boot, y_pred))


		ind = np.random.randint(x_test_imp_scal.shape[0], size=x_test_imp_scal.shape[0])
		x_boot = x_test_imp_scal[ind]
		y_boot = y_test[ind]

		y_pred = logReg.predict(x_boot)
		tn, fp, fn, tp = confusion_matrix(y_boot, y_pred).ravel()
		SENS_test_boot.append(tp / (tp + fn)) 
		SPEC_test_boot.append(tn / (tn + fp)) 
		PPV_test_boot.append(tp / (tp + fp)) 
		MCC_test_boot.append(matthews_corrcoef(y_boot, y_pred))






	# apply to test and get statistics
	logging.info("TRAIN")
	y_pred = logReg.predict(x_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_boot, 0.025), ", ", np.quantile(SENS_train_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_boot, 0.025), ", ", np.quantile(SPEC_train_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_boot, 0.025), ", ", np.quantile(PPV_train_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_train, y_pred), " (", np.quantile(MCC_train_boot, 0.025), ", ", np.quantile(MCC_train_boot, 0.975), ")")

	logging.info(" ")
	logging.info(" ")

	logging.info("TEST")
	y_pred = logReg.predict(x_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_boot, 0.025), ", ", np.quantile(SENS_test_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_boot, 0.025), ", ", np.quantile(SPEC_test_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_boot, 0.025), ", ", np.quantile(PPV_test_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_test, y_pred), " (", np.quantile(MCC_test_boot, 0.025), ", ", np.quantile(MCC_test_boot, 0.975), ")")

	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")











	### Model 2
	logging.info("Model 2")
	logging.info("Indel and SNV: CADD + AF + CSQ")

	x = df.filter(['CADD_PHRED', 'GERP', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x = df.filter(['CADD_PHRED', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])


	x_indel = x[ x['typeCV'] == 'indel' ]
	x_indel = x_indel.drop(['typeCV'], axis=1)
	x_indel = pd.get_dummies(x_indel, drop_first = True, columns = ['csqCV'])

	x_SNV = x[ x['typeCV'] == 'SNV' ]
	x_SNV = x_SNV.drop(['typeCV'], axis=1)
	x_SNV = pd.get_dummies(x_SNV, drop_first = True, columns = ['csqCV'])


	y = df.filter(['setCV', 'typeCV'])

	y_indel = y[ y['typeCV'] == 'indel' ]
	y_indel = y_indel.drop(['typeCV'], axis=1)
	y_indel = y_indel.values.reshape(-1,1)

	y_SNV = y[ y['typeCV'] == 'SNV' ]
	y_SNV = y_SNV.drop(['typeCV'], axis=1)
	y_SNV = y_SNV.values.reshape(-1,1)


	# split into training and testing set
	x_indel_train, x_indel_test, y_indel_train, y_indel_test = train_test_split(x_indel, y_indel, test_size = 0.2, random_state = 123)

	x_SNV_train, x_SNV_test, y_SNV_train, y_SNV_test = train_test_split(x_SNV, y_SNV, test_size = 0.2, random_state = 123)



	# impute the missing data on the training set
	logging.info("Impute the data")

	imp_indel = SimpleImputer(strategy = 'median')
	imp_indel.fit(x_indel_train)
	x_indel_train_imp = imp_indel.transform(x_indel_train)
	x_indel_test_imp = imp_indel.transform(x_indel_test)


	imp_SNV = SimpleImputer(strategy = 'median')
	imp_SNV.fit(x_SNV_train)
	x_SNV_train_imp = imp_SNV.transform(x_SNV_train)
	x_SNV_test_imp = imp_SNV.transform(x_SNV_test)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal_indel = MinMaxScaler(clip = 'true')
	scal_indel.fit(x_indel_train_imp)
	x_indel_train_imp_scal = scal_indel.transform(x_indel_train_imp)
	x_indel_test_imp_scal = scal_indel.transform(x_indel_test_imp)


	scal_SNV = MinMaxScaler(clip = 'true')
	scal_SNV.fit(x_SNV_train_imp)
	x_SNV_train_imp_scal = scal_SNV.transform(x_SNV_train_imp)
	x_SNV_test_imp_scal = scal_SNV.transform(x_SNV_test_imp)





	# run logistic regression
	logging.info("Run logistic regression")

	logReg_indel = LogisticRegression(penalty = 'none')
	logReg_indel.fit(x_indel_train_imp_scal, y_indel_train)

	print("Indel:")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_indel_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_indel_train_imp_scal, i) for i in range(len(x_indel_train.columns))]
	vif_data["Coefficient"] = logReg_indel.coef_.flatten()
	print(vif_data)


	logReg_SNV = LogisticRegression(penalty = 'none')
	logReg_SNV.fit(x_SNV_train_imp_scal, y_SNV_train)

	print("\n\nSNV")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_SNV_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_SNV_train_imp_scal, i) for i in range(len(x_SNV_train.columns))]
	vif_data["Coefficient"] = logReg_SNV.coef_.flatten()
	print(vif_data)
	


	SENS_train_indel_boot = []
	SPEC_train_indel_boot = []
	PPV_train_indel_boot = []
	MCC_train_indel_boot = []

	SENS_train_SNV_boot = []
	SPEC_train_SNV_boot = []
	PPV_train_SNV_boot = []
	MCC_train_SNV_boot = []

	SENS_train_COMBINED_boot = []
	SPEC_train_COMBINED_boot = []
	PPV_train_COMBINED_boot = []
	MCC_train_COMBINED_boot = []


	SENS_test_indel_boot = []
	SPEC_test_indel_boot = []
	PPV_test_indel_boot = []
	MCC_test_indel_boot = []

	SENS_test_SNV_boot = []
	SPEC_test_SNV_boot = []
	PPV_test_SNV_boot = []
	MCC_test_SNV_boot = []

	SENS_test_COMBINED_boot = []
	SPEC_test_COMBINED_boot = []
	PPV_test_COMBINED_boot = []
	MCC_test_COMBINED_boot = []



	# boostrap the metrics
	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		# indel
		ind_indel = np.random.randint(x_indel_train_imp_scal.shape[0], size=x_indel_train_imp_scal.shape[0])
		x_indel_boot = x_indel_train_imp_scal[ind_indel]
		y_indel_boot = y_indel_train[ind_indel]

		y_indel_pred = logReg_indel.predict(x_indel_boot)
		tn, fp, fn, tp = confusion_matrix(y_indel_boot, y_indel_pred).ravel()
		SENS_train_indel_boot.append(tp / (tp + fn)) 
		SPEC_train_indel_boot.append(tn / (tn + fp)) 
		PPV_train_indel_boot.append(tp / (tp + fp)) 
		MCC_train_indel_boot.append(matthews_corrcoef(y_indel_boot, y_indel_pred))

	
		# SNV
		ind_SNV = np.random.randint(x_SNV_train_imp_scal.shape[0], size=x_SNV_train_imp_scal.shape[0])
		x_SNV_boot = x_SNV_train_imp_scal[ind_SNV]
		y_SNV_boot = y_SNV_train[ind_SNV]

		y_SNV_pred = logReg_SNV.predict(x_SNV_boot)
		tn, fp, fn, tp = confusion_matrix(y_SNV_boot, y_SNV_pred).ravel()
		SENS_train_SNV_boot.append(tp / (tp + fn)) 
		SPEC_train_SNV_boot.append(tn / (tn + fp)) 
		PPV_train_SNV_boot.append(tp / (tp + fp)) 
		MCC_train_SNV_boot.append(matthews_corrcoef(y_SNV_boot, y_SNV_pred))


		# COMBINED = indel + SNV
		y_COMBINED_pred = np.append(y_indel_pred, y_SNV_pred)
		y_COMBINED_boot = np.append(y_indel_boot, y_SNV_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_train_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_train_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_train_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_train_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))



		# indel
		ind_indel = np.random.randint(x_indel_test_imp_scal.shape[0], size=x_indel_test_imp_scal.shape[0])
		x_indel_boot = x_indel_test_imp_scal[ind_indel]
		y_indel_boot = y_indel_test[ind_indel]

		y_indel_pred = logReg_indel.predict(x_indel_boot)
		tn, fp, fn, tp = confusion_matrix(y_indel_boot, y_indel_pred).ravel()
		SENS_test_indel_boot.append(tp / (tp + fn)) 
		SPEC_test_indel_boot.append(tn / (tn + fp)) 
		PPV_test_indel_boot.append(tp / (tp + fp)) 
		MCC_test_indel_boot.append(matthews_corrcoef(y_indel_boot, y_indel_pred))

	
		# SNV
		ind_SNV = np.random.randint(x_SNV_test_imp_scal.shape[0], size=x_SNV_test_imp_scal.shape[0])
		x_SNV_boot = x_SNV_test_imp_scal[ind_SNV]
		y_SNV_boot = y_SNV_test[ind_SNV]

		y_SNV_pred = logReg_SNV.predict(x_SNV_boot)
		tn, fp, fn, tp = confusion_matrix(y_SNV_boot, y_SNV_pred).ravel()
		SENS_test_SNV_boot.append(tp / (tp + fn)) 
		SPEC_test_SNV_boot.append(tn / (tn + fp)) 
		PPV_test_SNV_boot.append(tp / (tp + fp)) 
		MCC_test_SNV_boot.append(matthews_corrcoef(y_SNV_boot, y_SNV_pred))


		# COMBINED = indel + SNV
		y_COMBINED_pred = np.append(y_indel_pred, y_SNV_pred)
		y_COMBINED_boot = np.append(y_indel_boot, y_SNV_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_test_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_test_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_test_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_test_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))






	# apply to test and get statistics
	logging.info("TRAIN")
	logging.info("Results: indel")
	y_indel_train_pred = logReg_indel.predict(x_indel_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_indel_train, y_indel_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_indel_boot, 0.025), ", ", np.quantile(SENS_train_indel_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_indel_boot, 0.025), ", ", np.quantile(SPEC_train_indel_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_indel_boot, 0.025), ", ", np.quantile(PPV_train_indel_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_indel_train, y_indel_train_pred), " (", np.quantile(MCC_train_indel_boot, 0.025), ", ", np.quantile(MCC_train_indel_boot, 0.975), ")")
	logging.info(" ")



	# SNV
	logging.info("Results: SNV")
	y_SNV_train_pred = logReg_SNV.predict(x_SNV_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_SNV_train, y_SNV_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_SNV_boot, 0.025), ", ", np.quantile(SENS_train_SNV_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_SNV_boot, 0.025), ", ", np.quantile(SPEC_train_SNV_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_SNV_boot, 0.025), ", ", np.quantile(PPV_train_SNV_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_SNV_train, y_SNV_train_pred), " (", np.quantile(MCC_train_SNV_boot, 0.025), ", ", np.quantile(MCC_train_SNV_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = indel + SNV
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_indel_train, y_SNV_train)
	y_COMBINED_pred = np.append(y_indel_train_pred, y_SNV_train_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_COMBINED_boot, 0.025), ", ", np.quantile(SENS_train_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_train_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_COMBINED_boot, 0.025), ", ", np.quantile(PPV_train_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_train_COMBINED_boot, 0.025), ", ", np.quantile(MCC_train_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")

	logging.info("TEST")

	# indel
	logging.info("Results: indel")
	y_indel_test_pred = logReg_indel.predict(x_indel_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_indel_test, y_indel_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_indel_boot, 0.025), ", ", np.quantile(SENS_test_indel_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_indel_boot, 0.025), ", ", np.quantile(SPEC_test_indel_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_indel_boot, 0.025), ", ", np.quantile(PPV_test_indel_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_indel_test, y_indel_test_pred), " (", np.quantile(MCC_test_indel_boot, 0.025), ", ", np.quantile(MCC_test_indel_boot, 0.975), ")")
	logging.info(" ")



	# SNV
	logging.info("Results: SNV")
	y_SNV_test_pred = logReg_SNV.predict(x_SNV_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_SNV_test, y_SNV_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_SNV_boot, 0.025), ", ", np.quantile(SENS_test_SNV_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_SNV_boot, 0.025), ", ", np.quantile(SPEC_test_SNV_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_SNV_boot, 0.025), ", ", np.quantile(PPV_test_SNV_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_SNV_test, y_SNV_test_pred), " (", np.quantile(MCC_test_SNV_boot, 0.025), ", ", np.quantile(MCC_test_SNV_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = indel + SNV
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_indel_test, y_SNV_test)
	y_COMBINED_pred = np.append(y_indel_test_pred, y_SNV_test_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_COMBINED_boot, 0.025), ", ", np.quantile(SENS_test_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_test_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_COMBINED_boot, 0.025), ", ", np.quantile(PPV_test_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_test_COMBINED_boot, 0.025), ", ", np.quantile(MCC_test_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")










	### Model 3 
	logging.info("Model 3")
	logging.info("Missense (AF + 5_PRED) and non-missense (CADD + AF + CSQ + TYPE)")

	x = df.filter(['CADD_PHRED', 'GERP', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x = df.filter(['CADD_PHRED', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])


	y = df.filter(['setCV', 'csqCV'])

	# split into training and testing set
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)


	
	# split into missense and non-missense
	x_train_missense = x_train[ x_train['csqCV'] == 'missense_variant' ]
	x_train_missense = x_train_missense.drop(['CADD_PHRED', 'typeCV', 'csqCV'], axis=1)

	x_test_missense = x_test[ x_test['csqCV'] == 'missense_variant' ]
	x_test_missense = x_test_missense.drop(['CADD_PHRED', 'typeCV', 'csqCV'], axis=1)


	x_train_nonMissense = x_train[ x_train['csqCV'] != 'missense_variant' ]
	x_train_nonMissense = x_train_nonMissense.drop(['MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score'], axis=1)

	x_test_nonMissense = x_test[ x_test['csqCV'] != 'missense_variant' ]
	x_test_nonMissense['csqCV'] = pd.Categorical(x_test_nonMissense['csqCV'], categories = sorted(x_train_nonMissense['csqCV'].unique()))
	x_test_nonMissense['typeCV'] = pd.Categorical(x_test_nonMissense['typeCV'], categories = sorted(x_train_nonMissense['typeCV'].unique()))
	x_test_nonMissense = x_test_nonMissense.drop(['MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score'], axis=1)

	x_train_nonMissense = pd.get_dummies(x_train_nonMissense, drop_first = True, columns = ['csqCV', 'typeCV'])
	x_test_nonMissense = pd.get_dummies(x_test_nonMissense, drop_first = True, columns = ['csqCV', 'typeCV'])



	y_train_missense = y_train[ y_train['csqCV'] == 'missense_variant' ]
	y_train_missense = y_train_missense.drop(['csqCV'], axis=1)
	y_train_missense = y_train_missense.values.reshape(-1,1)

	y_test_missense = y_test[ y_test['csqCV'] == 'missense_variant' ]
	y_test_missense = y_test_missense.drop(['csqCV'], axis=1)
	y_test_missense = y_test_missense.values.reshape(-1,1)


	y_train_nonMissense = y_train[ y_train['csqCV'] != 'missense_variant' ]
	y_train_nonMissense = y_train_nonMissense.drop(['csqCV'], axis=1)
	y_train_nonMissense = y_train_nonMissense.values.reshape(-1,1)

	y_test_nonMissense = y_test[ y_test['csqCV'] != 'missense_variant' ]
	y_test_nonMissense = y_test_nonMissense.drop(['csqCV'], axis=1)
	y_test_nonMissense = y_test_nonMissense.values.reshape(-1,1)






	# impute the missing data on the training set
	logging.info("Impute the data")

	imp_missense = SimpleImputer(strategy = 'median')
	imp_missense.fit(x_train_missense)
	x_train_missense_imp = imp_missense.transform(x_train_missense)
	x_test_missense_imp = imp_missense.transform(x_test_missense)


	imp_nonMissense = SimpleImputer(strategy = 'median')
	imp_nonMissense.fit(x_train_nonMissense)
	x_train_nonMissense_imp = imp_nonMissense.transform(x_train_nonMissense)
	x_test_nonMissense_imp = imp_nonMissense.transform(x_test_nonMissense)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal_missense = MinMaxScaler(clip = 'true')
	scal_missense.fit(x_train_missense_imp)
	x_train_missense_imp_scal = scal_missense.transform(x_train_missense_imp)
	x_test_missense_imp_scal = scal_missense.transform(x_test_missense_imp)


	scal_nonMissense = MinMaxScaler(clip = 'true')
	scal_nonMissense.fit(x_train_nonMissense_imp)
	x_train_nonMissense_imp_scal = scal_nonMissense.transform(x_train_nonMissense_imp)
	x_test_nonMissense_imp_scal = scal_nonMissense.transform(x_test_nonMissense_imp)




	# run logistic regression
	logging.info("Run logistic regression")

	logReg_missense = LogisticRegression(penalty = 'none')
	logReg_missense.fit(x_train_missense_imp_scal, y_train_missense)

	print("Missense")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_train_missense.columns
	vif_data["VIF"] = [variance_inflation_factor(x_train_missense_imp_scal, i) for i in range(len(x_train_missense.columns))]
	vif_data["Coefficient"] = logReg_missense.coef_.flatten()
	print(vif_data)



	logReg_nonMissense = LogisticRegression(penalty = 'none')
	logReg_nonMissense.fit(x_train_nonMissense_imp_scal, y_train_nonMissense)

	print("\n\nNon-Missense")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_train_nonMissense.columns
	vif_data["VIF"] = [variance_inflation_factor(x_train_nonMissense_imp_scal, i) for i in range(len(x_train_nonMissense.columns))]
	vif_data["Coefficient"] = logReg_nonMissense.coef_.flatten()
	print(vif_data)



	SENS_train_missense_boot = []
	SPEC_train_missense_boot = []
	PPV_train_missense_boot = []
	MCC_train_missense_boot = []

	SENS_train_nonMissense_boot = []
	SPEC_train_nonMissense_boot = []
	PPV_train_nonMissense_boot = []
	MCC_train_nonMissense_boot = []

	SENS_train_COMBINED_boot = []
	SPEC_train_COMBINED_boot = []
	PPV_train_COMBINED_boot = []
	MCC_train_COMBINED_boot = []


	SENS_test_missense_boot = []
	SPEC_test_missense_boot = []
	PPV_test_missense_boot = []
	MCC_test_missense_boot = []

	SENS_test_nonMissense_boot = []
	SPEC_test_nonMissense_boot = []
	PPV_test_nonMissense_boot = []
	MCC_test_nonMissense_boot = []

	SENS_test_COMBINED_boot = []
	SPEC_test_COMBINED_boot = []
	PPV_test_COMBINED_boot = []
	MCC_test_COMBINED_boot = []



	# boostrap the metrics
	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		# missense
		ind_missense = np.random.randint(x_train_missense_imp_scal.shape[0], size=x_train_missense_imp_scal.shape[0])
		x_missense_boot = x_train_missense_imp_scal[ind_missense]
		y_missense_boot = y_train_missense[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_train_missense_boot.append(tp / (tp + fn)) 
		SPEC_train_missense_boot.append(tn / (tn + fp)) 
		PPV_train_missense_boot.append(tp / (tp + fp)) 
		MCC_train_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissense
		ind_nonMissense = np.random.randint(x_train_nonMissense_imp_scal.shape[0], size=x_train_nonMissense_imp_scal.shape[0])
		x_nonMissense_boot = x_train_nonMissense_imp_scal[ind_nonMissense]
		y_nonMissense_boot = y_train_nonMissense[ind_nonMissense]

		y_nonMissense_pred = logReg_nonMissense.predict(x_nonMissense_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissense_boot, y_nonMissense_pred).ravel()
		SENS_train_nonMissense_boot.append(tp / (tp + fn)) 
		SPEC_train_nonMissense_boot.append(tn / (tn + fp)) 
		PPV_train_nonMissense_boot.append(tp / (tp + fp)) 
		MCC_train_nonMissense_boot.append(matthews_corrcoef(y_nonMissense_boot, y_nonMissense_pred))


		# COMBINED = missense + nonMissense
		y_COMBINED_pred = np.append(y_missense_pred, y_nonMissense_pred)
		y_COMBINED_boot = np.append(y_missense_boot, y_nonMissense_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_train_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_train_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_train_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_train_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))



		# missense
		ind_missense = np.random.randint(x_test_missense_imp_scal.shape[0], size=x_test_missense_imp_scal.shape[0])
		x_missense_boot = x_test_missense_imp_scal[ind_missense]
		y_missense_boot = y_test_missense[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_test_missense_boot.append(tp / (tp + fn)) 
		SPEC_test_missense_boot.append(tn / (tn + fp)) 
		PPV_test_missense_boot.append(tp / (tp + fp)) 
		MCC_test_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissense
		ind_nonMissense = np.random.randint(x_test_nonMissense_imp_scal.shape[0], size=x_test_nonMissense_imp_scal.shape[0])
		x_nonMissense_boot = x_test_nonMissense_imp_scal[ind_nonMissense]
		y_nonMissense_boot = y_test_nonMissense[ind_nonMissense]

		y_nonMissense_pred = logReg_nonMissense.predict(x_nonMissense_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissense_boot, y_nonMissense_pred).ravel()
		SENS_test_nonMissense_boot.append(tp / (tp + fn)) 
		SPEC_test_nonMissense_boot.append(tn / (tn + fp)) 
		PPV_test_nonMissense_boot.append(tp / (tp + fp)) 
		MCC_test_nonMissense_boot.append(matthews_corrcoef(y_nonMissense_boot, y_nonMissense_pred))


		# COMBINED = missense + nonMissense
		y_COMBINED_pred = np.append(y_missense_pred, y_nonMissense_pred)
		y_COMBINED_boot = np.append(y_missense_boot, y_nonMissense_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_test_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_test_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_test_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_test_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))





	# apply to test and get statistics
	logging.info("TRAIN")

	# missense
	logging.info("Results: missense")
	y_train_missense_pred = logReg_missense.predict(x_train_missense_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_train_missense, y_train_missense_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_missense_boot, 0.025), ", ", np.quantile(SENS_train_missense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_missense_boot, 0.025), ", ", np.quantile(SPEC_train_missense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_missense_boot, 0.025), ", ", np.quantile(PPV_train_missense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_train_missense, y_train_missense_pred), " (", np.quantile(MCC_train_missense_boot, 0.025), ", ", np.quantile(MCC_train_missense_boot, 0.975), ")")
	logging.info(" ")



	# nonMissense
	logging.info("Results: nonMissense")
	y_train_nonMissense_pred = logReg_nonMissense.predict(x_train_nonMissense_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_train_nonMissense, y_train_nonMissense_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_nonMissense_boot, 0.025), ", ", np.quantile(SENS_train_nonMissense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_nonMissense_boot, 0.025), ", ", np.quantile(SPEC_train_nonMissense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_nonMissense_boot, 0.025), ", ", np.quantile(PPV_train_nonMissense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_train_nonMissense, y_train_nonMissense_pred), " (", np.quantile(MCC_train_nonMissense_boot, 0.025), ", ", np.quantile(MCC_train_nonMissense_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = missense + nonMissense
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_train_missense, y_train_nonMissense)
	y_COMBINED_pred = np.append(y_train_missense_pred, y_train_nonMissense_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_COMBINED_boot, 0.025), ", ", np.quantile(SENS_train_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_train_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_COMBINED_boot, 0.025), ", ", np.quantile(PPV_train_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_train_COMBINED_boot, 0.025), ", ", np.quantile(MCC_train_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info("TEST")

	# missense
	logging.info("Results: missense")
	y_test_missense_pred = logReg_missense.predict(x_test_missense_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_test_missense, y_test_missense_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_missense_boot, 0.025), ", ", np.quantile(SENS_test_missense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_missense_boot, 0.025), ", ", np.quantile(SPEC_test_missense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_missense_boot, 0.025), ", ", np.quantile(PPV_test_missense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_test_missense, y_test_missense_pred), " (", np.quantile(MCC_test_missense_boot, 0.025), ", ", np.quantile(MCC_test_missense_boot, 0.975), ")")
	logging.info(" ")



	# nonMissense
	logging.info("Results: nonMissense")
	y_test_nonMissense_pred = logReg_nonMissense.predict(x_test_nonMissense_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_test_nonMissense, y_test_nonMissense_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_nonMissense_boot, 0.025), ", ", np.quantile(SENS_test_nonMissense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_nonMissense_boot, 0.025), ", ", np.quantile(SPEC_test_nonMissense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_nonMissense_boot, 0.025), ", ", np.quantile(PPV_test_nonMissense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_test_nonMissense, y_test_nonMissense_pred), " (", np.quantile(MCC_test_nonMissense_boot, 0.025), ", ", np.quantile(MCC_test_nonMissense_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = missense + nonMissense
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_test_missense, y_test_nonMissense)
	y_COMBINED_pred = np.append(y_test_missense_pred, y_test_nonMissense_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_COMBINED_boot, 0.025), ", ", np.quantile(SENS_test_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_test_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_COMBINED_boot, 0.025), ", ", np.quantile(PPV_test_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_test_COMBINED_boot, 0.025), ", ", np.quantile(MCC_test_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")













	### Model 4
	logging.info("Model 4")
	logging.info("Missense (CADD + AF) and non-missense (CADD + AF + CSQ + TYPE)")

	x = df.filter(['CADD_PHRED', 'GERP', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x = df.filter(['CADD_PHRED', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])


	x_missense = x[ x['csqCV'] == 'missense_variant' ]
	x_missense = x_missense.drop(['typeCV', 'csqCV'], axis=1)

	x_nonMissense = x[ x['csqCV'] != 'missense_variant' ]
	x_nonMissense = pd.get_dummies(x_nonMissense, drop_first = True, columns = ['csqCV', 'typeCV'])


	y = df.filter(['setCV', 'csqCV'])

	y_missense = y[ y['csqCV'] == 'missense_variant' ]
	y_missense = y_missense.drop(['csqCV'], axis=1)
	y_missense = y_missense.values.reshape(-1,1)

	y_nonMissense = y[ y['csqCV'] != 'missense_variant' ]
	y_nonMissense = y_nonMissense.drop(['csqCV'], axis=1)
	y_nonMissense = y_nonMissense.values.reshape(-1,1)


	# split into training and testing set
	x_missense_train, x_missense_test, y_missense_train, y_missense_test = train_test_split(x_missense, y_missense, test_size = 0.2, random_state = 123)

	x_nonMissense_train, x_nonMissense_test, y_nonMissense_train, y_nonMissense_test = train_test_split(x_nonMissense, y_nonMissense, test_size = 0.2, random_state = 123)



	# impute the missing data on the training set
	logging.info("Impute the data")

	imp_missense = SimpleImputer(strategy = 'median')
	imp_missense.fit(x_missense_train)
	x_missense_train_imp = imp_missense.transform(x_missense_train)
	x_missense_test_imp = imp_missense.transform(x_missense_test)


	imp_nonMissense = SimpleImputer(strategy = 'median')
	imp_nonMissense.fit(x_nonMissense_train)
	x_nonMissense_train_imp = imp_nonMissense.transform(x_nonMissense_train)
	x_nonMissense_test_imp = imp_nonMissense.transform(x_nonMissense_test)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal_missense = MinMaxScaler(clip = 'true')
	scal_missense.fit(x_missense_train_imp)
	x_missense_train_imp_scal = scal_missense.transform(x_missense_train_imp)
	x_missense_test_imp_scal = scal_missense.transform(x_missense_test_imp)


	scal_nonMissense = MinMaxScaler(clip = 'true')
	scal_nonMissense.fit(x_nonMissense_train_imp)
	x_nonMissense_train_imp_scal = scal_nonMissense.transform(x_nonMissense_train_imp)
	x_nonMissense_test_imp_scal = scal_nonMissense.transform(x_nonMissense_test_imp)





	# run logistic regression
	logging.info("Run logistic regression")

	logReg_missense = LogisticRegression(penalty = 'none')
	logReg_missense.fit(x_missense_train_imp_scal, y_missense_train)

	print("Missense")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_missense_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_missense_train_imp_scal, i) for i in range(len(x_missense_train.columns))]
	vif_data["Coefficient"] = logReg_missense.coef_.flatten()
	print(vif_data)


	logReg_nonMissense = LogisticRegression(penalty = 'none')
	logReg_nonMissense.fit(x_nonMissense_train_imp_scal, y_nonMissense_train)

	print("\n\nNon-missense")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_train_nonMissense.columns
	vif_data["VIF"] = [variance_inflation_factor(x_train_nonMissense_imp_scal, i) for i in range(len(x_train_nonMissense.columns))]
	vif_data["Coefficient"] = logReg_nonMissense.coef_.flatten()
	print(vif_data)


	SENS_train_missense_boot = []
	SPEC_train_missense_boot = []
	PPV_train_missense_boot = []
	MCC_train_missense_boot = []

	SENS_train_nonMissense_boot = []
	SPEC_train_nonMissense_boot = []
	PPV_train_nonMissense_boot = []
	MCC_train_nonMissense_boot = []

	SENS_train_COMBINED_boot = []
	SPEC_train_COMBINED_boot = []
	PPV_train_COMBINED_boot = []
	MCC_train_COMBINED_boot = []


	SENS_test_missense_boot = []
	SPEC_test_missense_boot = []
	PPV_test_missense_boot = []
	MCC_test_missense_boot = []

	SENS_test_nonMissense_boot = []
	SPEC_test_nonMissense_boot = []
	PPV_test_nonMissense_boot = []
	MCC_test_nonMissense_boot = []

	SENS_test_COMBINED_boot = []
	SPEC_test_COMBINED_boot = []
	PPV_test_COMBINED_boot = []
	MCC_test_COMBINED_boot = []



	# boostrap the metrics
	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		# missense
		ind_missense = np.random.randint(x_missense_train_imp_scal.shape[0], size=x_missense_train_imp_scal.shape[0])
		x_missense_boot = x_missense_train_imp_scal[ind_missense]
		y_missense_boot = y_missense_train[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_train_missense_boot.append(tp / (tp + fn)) 
		SPEC_train_missense_boot.append(tn / (tn + fp)) 
		PPV_train_missense_boot.append(tp / (tp + fp)) 
		MCC_train_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissense
		ind_nonMissense = np.random.randint(x_nonMissense_train_imp_scal.shape[0], size=x_nonMissense_train_imp_scal.shape[0])
		x_nonMissense_boot = x_nonMissense_train_imp_scal[ind_nonMissense]
		y_nonMissense_boot = y_nonMissense_train[ind_nonMissense]

		y_nonMissense_pred = logReg_nonMissense.predict(x_nonMissense_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissense_boot, y_nonMissense_pred).ravel()
		SENS_train_nonMissense_boot.append(tp / (tp + fn)) 
		SPEC_train_nonMissense_boot.append(tn / (tn + fp)) 
		PPV_train_nonMissense_boot.append(tp / (tp + fp)) 
		MCC_train_nonMissense_boot.append(matthews_corrcoef(y_nonMissense_boot, y_nonMissense_pred))


		# COMBINED = missense + nonMissense
		y_COMBINED_pred = np.append(y_missense_pred, y_nonMissense_pred)
		y_COMBINED_boot = np.append(y_missense_boot, y_nonMissense_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_train_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_train_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_train_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_train_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))


		# missense
		ind_missense = np.random.randint(x_missense_test_imp_scal.shape[0], size=x_missense_test_imp_scal.shape[0])
		x_missense_boot = x_missense_test_imp_scal[ind_missense]
		y_missense_boot = y_missense_test[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_test_missense_boot.append(tp / (tp + fn)) 
		SPEC_test_missense_boot.append(tn / (tn + fp)) 
		PPV_test_missense_boot.append(tp / (tp + fp)) 
		MCC_test_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissense
		ind_nonMissense = np.random.randint(x_nonMissense_test_imp_scal.shape[0], size=x_nonMissense_test_imp_scal.shape[0])
		x_nonMissense_boot = x_nonMissense_test_imp_scal[ind_nonMissense]
		y_nonMissense_boot = y_nonMissense_test[ind_nonMissense]

		y_nonMissense_pred = logReg_nonMissense.predict(x_nonMissense_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissense_boot, y_nonMissense_pred).ravel()
		SENS_test_nonMissense_boot.append(tp / (tp + fn)) 
		SPEC_test_nonMissense_boot.append(tn / (tn + fp)) 
		PPV_test_nonMissense_boot.append(tp / (tp + fp)) 
		MCC_test_nonMissense_boot.append(matthews_corrcoef(y_nonMissense_boot, y_nonMissense_pred))


		# COMBINED = missense + nonMissense
		y_COMBINED_pred = np.append(y_missense_pred, y_nonMissense_pred)
		y_COMBINED_boot = np.append(y_missense_boot, y_nonMissense_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_test_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_test_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_test_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_test_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))






	# apply to test and get statistics
	logging.info("TRAIN")
	
	# missense
	logging.info("Results: missense")
	y_missense_train_pred = logReg_missense.predict(x_missense_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_missense_train, y_missense_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_missense_boot, 0.025), ", ", np.quantile(SENS_train_missense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_missense_boot, 0.025), ", ", np.quantile(SPEC_train_missense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_missense_boot, 0.025), ", ", np.quantile(PPV_train_missense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_missense_train, y_missense_train_pred), " (", np.quantile(MCC_train_missense_boot, 0.025), ", ", np.quantile(MCC_train_missense_boot, 0.975), ")")
	logging.info(" ")



	# nonMissense
	logging.info("Results: nonMissense")
	y_nonMissense_train_pred = logReg_nonMissense.predict(x_nonMissense_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_nonMissense_train, y_nonMissense_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_nonMissense_boot, 0.025), ", ", np.quantile(SENS_train_nonMissense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_nonMissense_boot, 0.025), ", ", np.quantile(SPEC_train_nonMissense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_nonMissense_boot, 0.025), ", ", np.quantile(PPV_train_nonMissense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_nonMissense_train, y_nonMissense_train_pred), " (", np.quantile(MCC_train_nonMissense_boot, 0.025), ", ", np.quantile(MCC_train_nonMissense_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = missense + nonMissense
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_missense_train, y_nonMissense_train)
	y_COMBINED_pred = np.append(y_missense_train_pred, y_nonMissense_train_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_COMBINED_boot, 0.025), ", ", np.quantile(SENS_train_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_train_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_COMBINED_boot, 0.025), ", ", np.quantile(PPV_train_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_train_COMBINED_boot, 0.025), ", ", np.quantile(MCC_train_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info("TEST")
	
	# missense
	logging.info("Results: missense")
	y_missense_test_pred = logReg_missense.predict(x_missense_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_missense_test, y_missense_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_missense_boot, 0.025), ", ", np.quantile(SENS_test_missense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_missense_boot, 0.025), ", ", np.quantile(SPEC_test_missense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_missense_boot, 0.025), ", ", np.quantile(PPV_test_missense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_missense_test, y_missense_test_pred), " (", np.quantile(MCC_test_missense_boot, 0.025), ", ", np.quantile(MCC_test_missense_boot, 0.975), ")")
	logging.info(" ")



	# nonMissense
	logging.info("Results: nonMissense")
	y_nonMissense_test_pred = logReg_nonMissense.predict(x_nonMissense_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_nonMissense_test, y_nonMissense_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_nonMissense_boot, 0.025), ", ", np.quantile(SENS_test_nonMissense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_nonMissense_boot, 0.025), ", ", np.quantile(SPEC_test_nonMissense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_nonMissense_boot, 0.025), ", ", np.quantile(PPV_test_nonMissense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_nonMissense_test, y_nonMissense_test_pred), " (", np.quantile(MCC_test_nonMissense_boot, 0.025), ", ", np.quantile(MCC_test_nonMissense_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = missense + nonMissense
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(y_missense_test, y_nonMissense_test)
	y_COMBINED_pred = np.append(y_missense_test_pred, y_nonMissense_test_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_COMBINED_boot, 0.025), ", ", np.quantile(SENS_test_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_test_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_COMBINED_boot, 0.025), ", ", np.quantile(PPV_test_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_test_COMBINED_boot, 0.025), ", ", np.quantile(MCC_test_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")












	### Model 5
	logging.info("Model 5")
	logging.info("Missense (CADD + AF + 5_PRED), non-missense SNV (CADD + AF + CSQ) and indel (CADD + AF + CSQ)")

	x = df.filter(['CADD_PHRED', 'GERP', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])
	x = df.filter(['CADD_PHRED', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score', 'gnomAD_v2_exome_AF_popmax', 'csqCV', 'typeCV'])


	x_indel = x[ x['typeCV'] == 'indel' ]
	x_indel = x_indel.drop(['typeCV', 'MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score'], axis=1)
	x_indel = pd.get_dummies(x_indel, drop_first = True, columns = ['csqCV'])


	x_missense = x[ x['csqCV'] == 'missense_variant' ]
	x_missense = x_missense.drop(['CADD_PHRED', 'typeCV', 'csqCV'], axis=1)

	x_nonMissenseSNV = x[ (x['csqCV'] != 'missense_variant') & (x['typeCV'] == 'SNV') ]
	x_nonMissenseSNV = x_nonMissenseSNV.drop(['MPC_score', 'Polyphen2_HDIV_score', 'REVEL_score', 'SIFT_score', 'FATHMM_score', 'typeCV'], axis=1)
	x_nonMissenseSNV = pd.get_dummies(x_nonMissenseSNV, drop_first = True, columns = ['csqCV'])


	y = df.filter(['setCV', 'csqCV', 'typeCV'])

	y_indel = y[ y['typeCV'] == 'indel' ]
	y_indel = y_indel.drop(['typeCV', 'csqCV'], axis=1)
	y_indel = y_indel.values.reshape(-1,1)

	y_missense = y[ y['csqCV'] == 'missense_variant' ]
	y_missense = y_missense.drop(['csqCV', 'typeCV'], axis=1)
	y_missense = y_missense.values.reshape(-1,1)

	y_nonMissenseSNV = y[ (y['csqCV'] != 'missense_variant') & (y['typeCV'] == 'SNV') ]
	y_nonMissenseSNV = y_nonMissenseSNV.drop(['csqCV', 'typeCV'], axis=1)
	y_nonMissenseSNV = y_nonMissenseSNV.values.reshape(-1,1)


	# split into training and testing set
	x_indel_train, x_indel_test, y_indel_train, y_indel_test = train_test_split(x_indel, y_indel, test_size = 0.2, random_state = 123)

	x_missense_train, x_missense_test, y_missense_train, y_missense_test = train_test_split(x_missense, y_missense, test_size = 0.2, random_state = 123)

	x_nonMissenseSNV_train, x_nonMissenseSNV_test, y_nonMissenseSNV_train, y_nonMissenseSNV_test = train_test_split(x_nonMissenseSNV, y_nonMissenseSNV, test_size = 0.2, random_state = 123)



	# impute the missing data on the training set
	logging.info("Impute the data")

	imp_indel = SimpleImputer(strategy = 'median', verbose = 100)
	imp_indel.fit(x_indel_train)
	x_indel_train_imp = imp_indel.transform(x_indel_train)
	x_indel_test_imp = imp_indel.transform(x_indel_test)


	imp_missense = SimpleImputer(strategy = 'median')
	imp_missense.fit(x_missense_train)
	x_missense_train_imp = imp_missense.transform(x_missense_train)
	x_missense_test_imp = imp_missense.transform(x_missense_test)


	imp_nonMissenseSNV = SimpleImputer(strategy = 'median', verbose = 100)
	imp_nonMissenseSNV.fit(x_nonMissenseSNV_train)
	x_nonMissenseSNV_train_imp = imp_nonMissenseSNV.transform(x_nonMissenseSNV_train)
	x_nonMissenseSNV_test_imp = imp_nonMissenseSNV.transform(x_nonMissenseSNV_test)



	# scale the data
	logging.info("Scaling to [0,1]")
	scal_indel = MinMaxScaler(clip = 'true')
	scal_indel.fit(x_indel_train_imp)
	x_indel_train_imp_scal = scal_indel.transform(x_indel_train_imp)
	x_indel_test_imp_scal = scal_indel.transform(x_indel_test_imp)


	scal_missense = MinMaxScaler(clip = 'true')
	scal_missense.fit(x_missense_train_imp)
	x_missense_train_imp_scal = scal_missense.transform(x_missense_train_imp)
	x_missense_test_imp_scal = scal_missense.transform(x_missense_test_imp)


	scal_nonMissenseSNV = MinMaxScaler(clip = 'true')
	scal_nonMissenseSNV.fit(x_nonMissenseSNV_train_imp)
	x_nonMissenseSNV_train_imp_scal = scal_nonMissenseSNV.transform(x_nonMissenseSNV_train_imp)
	x_nonMissenseSNV_test_imp_scal = scal_nonMissenseSNV.transform(x_nonMissenseSNV_test_imp)





	# run logistic regression
	logging.info("Run logistic regression")

	logReg_indel = LogisticRegression(penalty = 'none')
	logReg_indel.fit(x_indel_train_imp_scal, y_indel_train)

	print("Indel")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_indel_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_indel_train_imp_scal, i) for i in range(len(x_indel_train.columns))]
	vif_data["Coefficient"] = logReg_indel.coef_.flatten()
	print(vif_data)


	logReg_missense = LogisticRegression(penalty = 'none')
	logReg_missense.fit(x_missense_train_imp_scal, y_missense_train)

	print("\n\nMissense")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_missense_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_missense_train_imp_scal, i) for i in range(len(x_missense_train.columns))]
	vif_data["Coefficient"] = logReg_missense.coef_.flatten()
	print(vif_data)


	logReg_nonMissenseSNV = LogisticRegression(penalty = 'none')
	logReg_nonMissenseSNV.fit(x_nonMissenseSNV_train_imp_scal, y_nonMissenseSNV_train)

	print("\n\nNon-Missense SNV")
	vif_data = pd.DataFrame()
	vif_data["feature"] = x_nonMissenseSNV_train.columns
	vif_data["VIF"] = [variance_inflation_factor(x_nonMissenseSNV_train_imp_scal, i) for i in range(len(x_nonMissenseSNV_train.columns))]
	vif_data["Coefficient"] = logReg_nonMissenseSNV.coef_.flatten()
	print(vif_data)


	SENS_train_indel_boot = []
	SPEC_train_indel_boot = []
	PPV_train_indel_boot = []
	MCC_train_indel_boot = []

	SENS_train_missense_boot = []
	SPEC_train_missense_boot = []
	PPV_train_missense_boot = []
	MCC_train_missense_boot = []

	SENS_train_nonMissenseSNV_boot = []
	SPEC_train_nonMissenseSNV_boot = []
	PPV_train_nonMissenseSNV_boot = []
	MCC_train_nonMissenseSNV_boot = []

	SENS_train_COMBINED_boot = []
	SPEC_train_COMBINED_boot = []
	PPV_train_COMBINED_boot = []
	MCC_train_COMBINED_boot = []


	SENS_test_indel_boot = []
	SPEC_test_indel_boot = []
	PPV_test_indel_boot = []
	MCC_test_indel_boot = []

	SENS_test_missense_boot = []
	SPEC_test_missense_boot = []
	PPV_test_missense_boot = []
	MCC_test_missense_boot = []

	SENS_test_nonMissenseSNV_boot = []
	SPEC_test_nonMissenseSNV_boot = []
	PPV_test_nonMissenseSNV_boot = []
	MCC_test_nonMissenseSNV_boot = []

	SENS_test_COMBINED_boot = []
	SPEC_test_COMBINED_boot = []
	PPV_test_COMBINED_boot = []
	MCC_test_COMBINED_boot = []



	# boostrap the metrics
	logging.info("Bootstrap the performance metrics")
	for i in range(nBoot):
		# indel
		ind_indel = np.random.randint(x_indel_train_imp_scal.shape[0], size=x_indel_train_imp_scal.shape[0])
		x_indel_boot = x_indel_train_imp_scal[ind_indel]
		y_indel_boot = y_indel_train[ind_indel]

		y_indel_pred = logReg_indel.predict(x_indel_boot)
		tn, fp, fn, tp = confusion_matrix(y_indel_boot, y_indel_pred).ravel()
		SENS_train_indel_boot.append(tp / (tp + fn)) 
		SPEC_train_indel_boot.append(tn / (tn + fp)) 
		PPV_train_indel_boot.append(tp / (tp + fp)) 
		MCC_train_indel_boot.append(matthews_corrcoef(y_indel_boot, y_indel_pred))

	
		# missense
		ind_missense = np.random.randint(x_missense_train_imp_scal.shape[0], size=x_missense_train_imp_scal.shape[0])
		x_missense_boot = x_missense_train_imp_scal[ind_missense]
		y_missense_boot = y_missense_train[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_train_missense_boot.append(tp / (tp + fn)) 
		SPEC_train_missense_boot.append(tn / (tn + fp)) 
		PPV_train_missense_boot.append(tp / (tp + fp)) 
		MCC_train_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissenseSNV
		ind_nonMissenseSNV = np.random.randint(x_nonMissenseSNV_train_imp_scal.shape[0], size=x_nonMissenseSNV_train_imp_scal.shape[0])
		x_nonMissenseSNV_boot = x_nonMissenseSNV_train_imp_scal[ind_nonMissenseSNV]
		y_nonMissenseSNV_boot = y_nonMissenseSNV_train[ind_nonMissenseSNV]

		y_nonMissenseSNV_pred = logReg_nonMissenseSNV.predict(x_nonMissenseSNV_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissenseSNV_boot, y_nonMissenseSNV_pred).ravel()
		SENS_train_nonMissenseSNV_boot.append(tp / (tp + fn)) 
		SPEC_train_nonMissenseSNV_boot.append(tn / (tn + fp)) 
		PPV_train_nonMissenseSNV_boot.append(tp / (tp + fp)) 
		MCC_train_nonMissenseSNV_boot.append(matthews_corrcoef(y_nonMissenseSNV_boot, y_nonMissenseSNV_pred))


		# COMBINED = indel + missense + nonMissenseSNV
		y_COMBINED_pred = np.append(np.append(y_indel_pred, y_missense_pred), y_nonMissenseSNV_pred)
		y_COMBINED_boot = np.append(np.append(y_indel_boot, y_missense_boot), y_nonMissenseSNV_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_train_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_train_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_train_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_train_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))



		# indel
		ind_indel = np.random.randint(x_indel_test_imp_scal.shape[0], size=x_indel_test_imp_scal.shape[0])
		x_indel_boot = x_indel_test_imp_scal[ind_indel]
		y_indel_boot = y_indel_test[ind_indel]

		y_indel_pred = logReg_indel.predict(x_indel_boot)
		tn, fp, fn, tp = confusion_matrix(y_indel_boot, y_indel_pred).ravel()
		SENS_test_indel_boot.append(tp / (tp + fn)) 
		SPEC_test_indel_boot.append(tn / (tn + fp)) 
		PPV_test_indel_boot.append(tp / (tp + fp)) 
		MCC_test_indel_boot.append(matthews_corrcoef(y_indel_boot, y_indel_pred))

	
		# missense
		ind_missense = np.random.randint(x_missense_test_imp_scal.shape[0], size=x_missense_test_imp_scal.shape[0])
		x_missense_boot = x_missense_test_imp_scal[ind_missense]
		y_missense_boot = y_missense_test[ind_missense]

		y_missense_pred = logReg_missense.predict(x_missense_boot)
		tn, fp, fn, tp = confusion_matrix(y_missense_boot, y_missense_pred).ravel()
		SENS_test_missense_boot.append(tp / (tp + fn)) 
		SPEC_test_missense_boot.append(tn / (tn + fp)) 
		PPV_test_missense_boot.append(tp / (tp + fp)) 
		MCC_test_missense_boot.append(matthews_corrcoef(y_missense_boot, y_missense_pred))

	
		# nonMissenseSNV
		ind_nonMissenseSNV = np.random.randint(x_nonMissenseSNV_test_imp_scal.shape[0], size=x_nonMissenseSNV_test_imp_scal.shape[0])
		x_nonMissenseSNV_boot = x_nonMissenseSNV_test_imp_scal[ind_nonMissenseSNV]
		y_nonMissenseSNV_boot = y_nonMissenseSNV_test[ind_nonMissenseSNV]

		y_nonMissenseSNV_pred = logReg_nonMissenseSNV.predict(x_nonMissenseSNV_boot)
		tn, fp, fn, tp = confusion_matrix(y_nonMissenseSNV_boot, y_nonMissenseSNV_pred).ravel()
		SENS_test_nonMissenseSNV_boot.append(tp / (tp + fn)) 
		SPEC_test_nonMissenseSNV_boot.append(tn / (tn + fp)) 
		PPV_test_nonMissenseSNV_boot.append(tp / (tp + fp)) 
		MCC_test_nonMissenseSNV_boot.append(matthews_corrcoef(y_nonMissenseSNV_boot, y_nonMissenseSNV_pred))


		# COMBINED = indel + missense + nonMissenseSNV
		y_COMBINED_pred = np.append(np.append(y_indel_pred, y_missense_pred), y_nonMissenseSNV_pred)
		y_COMBINED_boot = np.append(np.append(y_indel_boot, y_missense_boot), y_nonMissenseSNV_boot)

		tn, fp, fn, tp = confusion_matrix(y_COMBINED_boot, y_COMBINED_pred).ravel()
		SENS_test_COMBINED_boot.append(tp / (tp + fn)) 
		SPEC_test_COMBINED_boot.append(tn / (tn + fp)) 
		PPV_test_COMBINED_boot.append(tp / (tp + fp)) 
		MCC_test_COMBINED_boot.append(matthews_corrcoef(y_COMBINED_boot, y_COMBINED_pred))






	# apply to test and get statistics
	logging.info("TRAIN")
	# indel
	logging.info("Results: indel")
	y_indel_train_pred = logReg_indel.predict(x_indel_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_indel_train, y_indel_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_indel_boot, 0.025), ", ", np.quantile(SENS_train_indel_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_indel_boot, 0.025), ", ", np.quantile(SPEC_train_indel_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_indel_boot, 0.025), ", ", np.quantile(PPV_train_indel_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_indel_train, y_indel_train_pred), " (", np.quantile(MCC_train_indel_boot, 0.025), ", ", np.quantile(MCC_train_indel_boot, 0.975), ")")
	logging.info(" ")



	# missense
	logging.info("Results: missense")
	y_missense_train_pred = logReg_missense.predict(x_missense_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_missense_train, y_missense_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_missense_boot, 0.025), ", ", np.quantile(SENS_train_missense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_missense_boot, 0.025), ", ", np.quantile(SPEC_train_missense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_missense_boot, 0.025), ", ", np.quantile(PPV_train_missense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_missense_train, y_missense_train_pred), " (", np.quantile(MCC_train_missense_boot, 0.025), ", ", np.quantile(MCC_train_missense_boot, 0.975), ")")
	logging.info(" ")



	# nonMissenseSNV
	logging.info("Results: nonMissenseSNV")
	y_nonMissenseSNV_train_pred = logReg_nonMissenseSNV.predict(x_nonMissenseSNV_train_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_nonMissenseSNV_train, y_nonMissenseSNV_train_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_nonMissenseSNV_boot, 0.025), ", ", np.quantile(SENS_train_nonMissenseSNV_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_nonMissenseSNV_boot, 0.025), ", ", np.quantile(SPEC_train_nonMissenseSNV_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_nonMissenseSNV_boot, 0.025), ", ", np.quantile(PPV_train_nonMissenseSNV_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_nonMissenseSNV_train, y_nonMissenseSNV_train_pred), " (", np.quantile(MCC_train_nonMissenseSNV_boot, 0.025), ", ", np.quantile(MCC_train_nonMissenseSNV_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = indel + missense + nonMissenseSNV
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(np.append(y_indel_train, y_missense_train), y_nonMissenseSNV_train)
	y_COMBINED_pred = np.append(np.append(y_indel_train_pred, y_missense_train_pred), y_nonMissenseSNV_train_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_train_COMBINED_boot, 0.025), ", ", np.quantile(SENS_train_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_train_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_train_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_train_COMBINED_boot, 0.025), ", ", np.quantile(PPV_train_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_train_COMBINED_boot, 0.025), ", ", np.quantile(MCC_train_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info("TEST")
	# indel
	logging.info("Results: indel")
	y_indel_test_pred = logReg_indel.predict(x_indel_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_indel_test, y_indel_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_indel_boot, 0.025), ", ", np.quantile(SENS_test_indel_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_indel_boot, 0.025), ", ", np.quantile(SPEC_test_indel_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_indel_boot, 0.025), ", ", np.quantile(PPV_test_indel_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_indel_test, y_indel_test_pred), " (", np.quantile(MCC_test_indel_boot, 0.025), ", ", np.quantile(MCC_test_indel_boot, 0.975), ")")
	logging.info(" ")



	# missense
	logging.info("Results: missense")
	y_missense_test_pred = logReg_missense.predict(x_missense_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_missense_test, y_missense_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_missense_boot, 0.025), ", ", np.quantile(SENS_test_missense_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_missense_boot, 0.025), ", ", np.quantile(SPEC_test_missense_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_missense_boot, 0.025), ", ", np.quantile(PPV_test_missense_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_missense_test, y_missense_test_pred), " (", np.quantile(MCC_test_missense_boot, 0.025), ", ", np.quantile(MCC_test_missense_boot, 0.975), ")")
	logging.info(" ")



	# nonMissenseSNV
	logging.info("Results: nonMissenseSNV")
	y_nonMissenseSNV_test_pred = logReg_nonMissenseSNV.predict(x_nonMissenseSNV_test_imp_scal)
	tn, fp, fn, tp = confusion_matrix(y_nonMissenseSNV_test, y_nonMissenseSNV_test_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_nonMissenseSNV_boot, 0.025), ", ", np.quantile(SENS_test_nonMissenseSNV_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_nonMissenseSNV_boot, 0.025), ", ", np.quantile(SPEC_test_nonMissenseSNV_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_nonMissenseSNV_boot, 0.025), ", ", np.quantile(PPV_test_nonMissenseSNV_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_nonMissenseSNV_test, y_nonMissenseSNV_test_pred), " (", np.quantile(MCC_test_nonMissenseSNV_boot, 0.025), ", ", np.quantile(MCC_test_nonMissenseSNV_boot, 0.975), ")")
	logging.info(" ")



	# COMBINED = indel + missense + nonMissenseSNV
	logging.info("Results: COMBINED")
	y_COMBINED = np.append(np.append(y_indel_test, y_missense_test), y_nonMissenseSNV_test)
	y_COMBINED_pred = np.append(np.append(y_indel_test_pred, y_missense_test_pred), y_nonMissenseSNV_test_pred)
	tn, fp, fn, tp = confusion_matrix(y_COMBINED, y_COMBINED_pred).ravel()
	
	print("\tSensitivity - ", tp / (tp + fn), " (", np.quantile(SENS_test_COMBINED_boot, 0.025), ", ", np.quantile(SENS_test_COMBINED_boot, 0.975), ")")
	print("\tSpecificity - ", tn / (tn + fp), " (", np.quantile(SPEC_test_COMBINED_boot, 0.025), ", ", np.quantile(SPEC_test_COMBINED_boot, 0.975), ")")
	print("\tPPV - ", tp / (tp + fp), " (", np.quantile(PPV_test_COMBINED_boot, 0.025), ", ", np.quantile(PPV_test_COMBINED_boot, 0.975), ")")
	print("\tMCC - ", matthews_corrcoef(y_COMBINED, y_COMBINED_pred), " (", np.quantile(MCC_test_COMBINED_boot, 0.025), ", ", np.quantile(MCC_test_COMBINED_boot, 0.975), ")")


	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")


















	logging.info("Done")

	




if __name__ == "__main__":
	main(sys.argv[1:])



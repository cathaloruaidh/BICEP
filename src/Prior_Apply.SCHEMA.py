#!/usr/bin/python




import cProfile, getopt, glob, logging, math, os, pickle, pprint, re, sys
import numpy as np
import pandas as pd

from cyvcf2 import VCF, Writer

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import confusion_matrix, matthews_corrcoef, classification_report, r2_score
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

from joblib import dump, load

import warnings





# main function
def main(argv):


	# ignore warnings
	warnings.filterwarnings("ignore")



	# command line arguments

	nCores = 1
	inputVcfFile = None
	logLevel = None
	modelPrefix = None
	outputPrefix = None

	try:
		opts, args = getopt.getopt(argv, "c:l:m:o:v:", ["cores=", "log=", "model=", "output=", "vcf="])
	except getopt.GetoptError:
		print("Getopt Error")
		logging.error("getopt error")
		sys.exit("Exiting ... ")

	for opt, arg in opts:
		if opt in ("-b", "--boot"):
			nBoot = int(arg)

		if opt in ("-c", "--cores"):
			if int(arg) <= multiprocessing.cpu_count():
				nCores = int(arg)

		if opt in ("-l", "--log"):
			logLevel = arg.upper()
			numericLevel = getattr(logging, arg.upper(), None)
			if not isinstance(numericLevel, int):
				raise ValueError('Invalid log level: %s' % arg)

		if opt in ("-m", "--model"):
			modelPrefix = arg
	
		if opt in ("-o", "--output"):
			outputPrefix = arg
	
		if opt in ("-v", "--vcf"):
			inputVcfFile = arg
	


	# get the output file names
	if outputPrefix is None:
		outputErr= "output.err"

	else:
		outputErr = outputPrefix + ".err"



	# set the logging info
	FORMAT = '# %(asctime)s [%(levelname)s] - %(message)s'
	
	if logLevel is None:
		logging.basicConfig(format=FORMAT)
	else:
		numericLevel = getattr(logging, logLevel, None)
		if not isinstance(numericLevel, int):
			raise ValueError('Invalid log level: %s' % logLevel)
		logging.basicConfig(format=FORMAT, level=logLevel)
	
	
	# add colours to the log name
	logging.addLevelName(logging.NOTSET, "NOT  ")
	logging.addLevelName(logging.DEBUG, "\u001b[36mDEBUG\u001b[0m")
	logging.addLevelName(logging.INFO, "INFO ")
	logging.addLevelName(logging.WARNING, "\u001b[33mWARN \u001b[0m")
	logging.addLevelName(logging.ERROR, "\u001b[31mERROR\u001b[0m")
	logging.addLevelName(logging.CRITICAL, "\u001b[35mCRIT\u001b[0m")


	# check if model files exist
	if modelPrefix is None:
		logging.error("No model files specified")
		sys.exit(1)

	if not glob.glob(modelPrefix + "*.pkl"):
		logging.error("No model files found")
		sys.exit(1)


	# define VEP hierarchy for ClinVar consequences examined
	vepCSQRank_CV = {'splice_acceptor_variant' : 1, 
	'splice_donor_variant' : 2, 
	'stop_gained' : 3, 
	'frameshift_variant' : 4, 
	'stop_lost' : 5, 
	'inframe_insertion' : 6, 
	'inframe_deletion' : 7, 
	'missense_variant' : 8, 
	'synonymous_variant' : 9, 
	'5_prime_UTR_variant' : 10, 
	'3_prime_UTR_variant' : 11, 
	'intron_variant' : 12 }

	vepCSQRank_SCHEMA = {'splice_acceptor_variant' : 1, 
	'splice_donor_variant' : 2, 
	'stop_gained' : 3, 
	'frameshift_variant' : 4, 
	'missense_variant' : 5} 




	################################################################################
	# set genotype information
	################################################################################

	logging.info("Reading VCF file")



	vcf = VCF(inputVcfFile, gts012=True)


	keys = re.sub('^.*?: ', '', vcf.get_header_type('CSQ')['Description']).split("|")
	keys = [ key.strip("\"") for key in keys ]
	keysPredictors = [ "MPC_score", "gnomAD_v2_exome_AF_popmax" ]
	



	# print header
	with open(outputErr, 'w') as f:
		print("ID\tINFO\tOTHER", file=f)



	# save variants in list 
	DATA = []
	for variant in vcf:
		# get the ID of the variant
		ID = variant.CHROM + "_" + str(variant.start+1) + "_" + variant.REF + "_" + variant.ALT[0] 

		msg = "Reading variant " + ID
		logging.debug(msg)



		# get the variant type, ignore if not SNV or indel
		if len(variant.REF) == 1 and len(variant.ALT[0]) == 1:
			typeVEP = "SNV"
		elif ( (len(variant.REF) == 1 and len(variant.ALT[0]) > 1) or ((len(variant.REF) > 1 and len(variant.ALT[0]) == 1)) ) and max(len(variant.REF), len(variant.ALT[0])) < 50:
			typeVEP = "indel"
		else:
			with open(outputErr, 'a') as f:
				print(ID, "\tNA\tTYPE", file=f)
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


			# select the consequence with the highest vep rank
			csqCVSubset = [ x for x in dictVEP["Consequence_split"] if x in vepCSQRank_CV.keys() ]
			
			if len(csqCVSubset) > 0:
				d = dict((k, vepCSQRank_CV[k]) for k in csqCVSubset)
				dictVEP["Consequence_select"] = min(d, key=d.get)
			else:
				dictVEP["Consequence_select"] = np.nan


			# extract the transcript-specific metrics from dbNSFP
			# and pick the most deleterious value
			for key in keysPredictors:
				l = [ float(x) for x in dictVEP[key].split("&") if x != '.' and x != "" ]
				if len(l) == 0:
					dictVEP[key] = np.nan
				else:
					dictVEP[key] = max(l)

			

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

				for key in keysPredictors:
					l.append(csqVEP[i][key])

				DATA.append(l) 


		# otherwise print to error file
		else:
			with open(outputErr, 'a') as f:
				print(ID, "\tNA\tNoTranscript", file=f)





	################################################################################
	# Apply the regression prediction
	################################################################################

	logging.info("Convert to Dataframe")

	pd.set_option('display.max_colwidth', 100)

	# read in the data
	df = pd.DataFrame(DATA, columns = ['ID', 'Gene', 'typeVEP', 'csqVEP'] + keysPredictors )

	df.to_csv("DATA.txt", index=False, sep='\t')

	df['csqVEP'] = pd.Categorical(df['csqVEP'], categories = sorted(vepCSQRank_CV.keys()))
	print(df['csqVEP'].value_counts())
	df['typeVEP'] = pd.Categorical(df['typeVEP'], categories = ['indel', 'SNV'])




	# set missing allele frequencies to zero
	#df["gnomAD_v2_exome_AF_popmax"] = df["gnomAD_v2_exome_AF_popmax"].fillna(0.0)


	### Model 3 
	logging.info("Model 3")
	logging.info("Missense (AF + MPC) and non-missense (AF + CSQ + TYPE)")

	x = df.filter(['MPC_score', 'gnomAD_v2_exome_AF_popmax', 'csqVEP', 'typeVEP'])
	

	x_missense = x[ x['csqVEP'] == 'missense_variant' ]
	x_missense_csq = x_missense['csqVEP']
	x_missense = x_missense.drop(['typeVEP', 'csqVEP'], axis=1)
	x_missense_index = x_missense.index


	x_nonMissense = x[ x['csqVEP'] != 'missense_variant' ]
	x_nonMissense_csq = x_nonMissense['csqVEP']
	x_nonMissense = pd.get_dummies(x_nonMissense, columns = ['csqVEP', 'typeVEP'])
	x_nonMissense = x_nonMissense.drop(['MPC_score', 'csqVEP_3_prime_UTR_variant', 'csqVEP_missense_variant', 'typeVEP_SNV'], axis=1)
	x_nonMissense_index = x_nonMissense.index


	# zero out consequences that don't form part of SCHEMA prioritisation
	logging.info("Zero out non-SCHEMA functional consequences")
	colsZero = [ "csqVEP_" + k for k in vepCSQRank_CV.keys() if k not in vepCSQRank_SCHEMA.keys() ]
	colsZero = [ k for k in colsZero if k in x_nonMissense.columns ]
	
	for col in colsZero:
		x_nonMissense[col].values[:] = 0.0





	y_missense = df[ df['csqVEP'] == 'missense_variant' ]

	y_nonMissense = df[ df['csqVEP'] != 'missense_variant' ]




	# impute the missing data
	logging.info("Impute the data")

	with open(modelPrefix + '.imp_missense.pkl', 'rb') as f:
		imp_missense = pickle.load(f)

	x_missense_imp = imp_missense.transform(x_missense)



	with open(modelPrefix + '.imp_nonMissense.pkl', 'rb') as f:
		imp_nonMissense = pickle.load(f)
	
	x_nonMissense_imp = imp_nonMissense.transform(x_nonMissense)



	# scale the data
	logging.info("Scaling to [0,1]")

	with open(modelPrefix + '.scal_missense.pkl', 'rb') as f:
		scal_missense = pickle.load(f)

	x_missense_imp_scal = scal_missense.transform(x_missense_imp)


	with open(modelPrefix + '.scal_nonMissense.pkl', 'rb') as f:
		scal_nonMissense = pickle.load(f)

	x_nonMissense_imp_scal = scal_nonMissense.transform(x_nonMissense_imp)




	# run logistic regression
	logging.info("Run logistic regression")

	with open(modelPrefix + '.logReg_missense.pkl', 'rb') as f:
		logReg_missense = pickle.load(f)


	with open(modelPrefix + '.logReg_nonMissense.pkl', 'rb') as f:
		logReg_nonMissense = pickle.load(f)


	#pprint.pprint(list(zip(np.sort(logReg_missense.feature_names), np.sort(x_missense.columns.values))))
	#print("\n")
	#pprint.pprint(list(zip(np.sort(logReg_nonMissense.feature_names), np.sort(x_nonMissense.columns.values))))


	pprint.pprint(list(zip(logReg_missense.feature_names, logReg_missense.coef_.flatten())))
	print("\n")
	pprint.pprint(list(zip(logReg_nonMissense.feature_names, logReg_nonMissense.coef_.flatten())))


	# missense
	y_missense_pred = logReg_missense.predict_proba(x_missense_imp_scal)


	# nonMissense
	y_nonMissense_pred = logReg_nonMissense.predict_proba(x_nonMissense_imp_scal)


	# combine and output to file
	prior_prob = pd.DataFrame()
	prior_prob["ID"] = np.append(y_missense["ID"], y_nonMissense["ID"]).flatten()
	prior_prob["csq"] = np.append(x_missense_csq, x_nonMissense_csq).flatten()
	prior_prob["Gene"] = np.append(y_missense["Gene"], y_nonMissense["Gene"]).flatten()
	prior_prob["prior"] = np.append(y_missense_pred[:,1], y_nonMissense_pred[:,1]).flatten()

	x_missense_imp_scal_df = pd.DataFrame(x_missense_imp_scal, index = x_missense_index, columns = x_missense.columns)
	x_nonMissense_imp_scal_df = pd.DataFrame(x_nonMissense_imp_scal, index = x_nonMissense_index, columns = x_nonMissense.columns)

	x_imp_scal_df = pd.concat([x_missense_imp_scal_df, x_nonMissense_imp_scal_df], ignore_index=True, sort=False)

	combined = pd.concat([x_imp_scal_df, prior_prob], axis=1)
	combined.to_csv(outputPrefix+".Model3.priors.txt", index=False, sep='\t', na_rep='.')



	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")
	logging.info(" ")









	logging.info("Done")

	




if __name__ == "__main__":
	main(sys.argv[1:])



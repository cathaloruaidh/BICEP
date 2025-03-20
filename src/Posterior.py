import csv
import logging
import math
import os
import plotly
import re
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import BayesFactor


#from plotly.subplots import make_subplots



# main function
def PO_main(args):

	logging.info("POSTERIOR")
	logging.info(" ")


	priorFile = None
	bfFile = None
	

	# command line arguments
	#if (args.prior is None) or (args.bf is None):
	priorFile = args.prefix + ".priors.txt"
	bfFile = args.prefix + ".BF.txt"

	#else:
	#	priorFile = args.prior
	#	bfFile = args.bf


	# load the prior and BF files
	logging.info("Reading in the prior file")
	with open(args.outputDir + priorFile) as f:
		prior = pd.read_csv(f, sep="\t", na_values=['.'])
		prior.dropna(subset=['prior'], inplace=True)
	
	
	logging.info("Reading in the Bayes factor file")
	with open(args.outputDir + bfFile) as f:
		bf = pd.read_csv(f, sep="\t", na_values=['.'], dtype = {"ID" : str, "BF" : float, "logBF" : float, "STRING" : str})
		bf.dropna(subset=['BF'], inplace=True)
		bf = bf[bf["BF"] > 0]
	


	# combine values and output
	logging.info("Merge and output")
	merged = prior.merge(bf, on="ID")


	# if Prior == 0, set to min non-zero
	merged.loc[merged["prior"] == 0, "prior"] = np.min(merged[merged["prior"] > 0]["prior"])
	merged["PriorOC"] = merged["prior"] / (1 - merged["prior"] )
	merged["logPriorOC"] = np.log10(merged["PriorOC"])

	# drop NA values
	merged.replace([np.inf, -np.inf], np.nan, inplace=True)
	merged.dropna(subset=['prior', 'PriorOC', 'logPriorOC', 'BF', 'logBF'], inplace=True)


	# calculate the posteriors
	merged["logPriorOC"] = pd.to_numeric(merged["logPriorOC"], errors='coerce')
	merged["logPostOC"] = merged["logPriorOC"] + merged["logBF"]
	merged.dropna(subset=['logPostOC'], inplace=True)
	merged["Rank"] = merged["logPostOC"].rank(ascending=False, method='first')


	# get rid of unnecessary columns and reorder
	merged = merged.drop(['prior', 'PriorOC', 'BF'], axis=1)

	if args.cnv:
		orderFirst = [ "Rank", "ID", "logPostOC", "logPriorOC", "logBF", "STRING" ]
	else:
		orderFirst = [ "Rank", "ID", "Gene", "csq", "impact", "logPostOC", "logPriorOC", "logBF", "STRING" ]

	orderSecond = [ x for x in merged.columns.tolist() if x not in orderFirst ]

	merged[ orderFirst ] = merged[ orderFirst ].round(3)
	merged = merged[ orderFirst + orderSecond ]
	merged = merged.sort_values("Rank")


	with open(args.tempDir + args.prefix + '.max_logBF.txt', 'r') as f:
		tmp = f.readlines()
		max_logBF = float(tmp[0])

	merged_sub = merged.sort_values(by=['logPostOC'], ascending=False).head(n=args.top)

	merged.to_csv(args.outputDir + args.prefix + ".posteriors.txt", index=False, sep='\t', na_rep='.')




	# plot top variants
	logging.info("Plotting the top " + str(args.top) +  " variants")

	matplotlib.use('agg')

	max_y = np.ceil(np.max(np.concatenate((merged_sub['logPostOC'].values, merged_sub['logBF'].values, merged_sub['logPriorOC'].values, np.array([max_logBF])))))
	min_y = np.floor(np.min(np.concatenate((merged_sub['logPostOC'].values, merged_sub['logBF'].values, merged_sub['logPriorOC'].values, np.array([max_logBF])))))

	# let the y_min be no larger than twice abs(y_max)
	# in case there are low-prior variants taking over the plot
	min_y = np.max([min_y, -2*abs(max_y)])


	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

	x_ticks = [1] + [ i for i in range(5, args.top + 1, 5) ]
	x_ticks_label = [ str(i) for i in x_ticks ]

	plt.setp((ax1, ax2, ax3), xticks=x_ticks, xticklabels=x_ticks_label, ylim=(min_y, max_y))
	plt.gca().set_ylim(min_y, max_y)

	if (args.highlight is not None) and (merged_sub["ID"].str.contains(args.highlight).any()):
		ax1.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logPostOC"], color="#61D04F", hatch='//')
		ax1.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logPostOC"], color="#61D04F")
	
	else:
		ax1.bar(merged_sub["Rank"], merged_sub["logPostOC"], color="#61D04F")
		
	ax1.set(ylabel="logPostOC")
	ax1.margins(0.05, 0.2)
	ax1.set_xlim([0, args.top + 1])
	ax1.axhline(y=0,linewidth=2, color='k')


	if (args.highlight is not None) and (merged_sub["ID"].str.contains(args.highlight).any()):
		ax2.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logBF"], color="#2297E6", hatch='//')
		ax2.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logBF"], color="#2297E6")

	else:
		ax2.bar(merged_sub["Rank"], merged_sub["logBF"], color="#2297E6")

	ax2.set(ylabel="logBF")
	ax2.margins(0.05, 0.2)
	ax2.set_xlim([0, args.top + 1])
	ax2.axhline(y=0,linewidth=2, color='k')
	ax2.axhline(y=max_logBF, linewidth=2, color='k', linestyle='--')


	if args.highlight is not None and (merged_sub["ID"].str.contains(args.highlight).any()):
		ax3.bar(merged_sub[merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[merged_sub["ID"].str.contains(args.highlight)]["logPriorOC"], color="#DF536B", hatch='//')
		ax3.bar(merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["Rank"], merged_sub[~merged_sub["ID"].str.contains(args.highlight)]["logPriorOC"], color="#DF536B")

	else:
		ax3.bar(merged_sub["Rank"], merged_sub["logPriorOC"], color="#DF536B")

	ax3.set(xlabel="Rank", ylabel="logPriorOC")
	ax3.margins(0.05, 0.2)
	ax3.set_xlim([0, args.top + 1])
	ax3.axhline(y=0,linewidth=2, color='k')

	plt.savefig(args.outputDir + args.prefix + ".BICEP.png", dpi=300)
	






	# get predictors
	if args.cnv:
		if args.predictors is not None:
	
			d = {}

			with open(args.predictors, 'r') as f:
				for line in f:
					(model, key)=line.split()
					d[model, key] = 1


			# manually add allele frequency to the predictors
			keysPredictors = [ x[1] for x in d.keys()] + [args.frequency]
			keysPredictors = sorted(list(set(keysPredictors)))


		else:
			keysPredictors = sorted([ "overlap_loeuf_sumRecip" ] + [ args.frequency ] )



	else:
		if args.predictors is not None:
	
			d = {}

			with open(args.predictors, 'r') as f:
				for line in f:
					(model, key, value)=line.split()
					d[model, key] = value


			# manually add allele frequency to the predictors
			keysPredictors = sorted([ x[1] for x in d.keys()] + [args.frequency])



		else:
			keysPredictors = [ "FATHMM_score", "MPC_score", "Polyphen2_HDIV_score", "REVEL_score", "SIFT_score" ] + [ args.frequency ] 


	# plotly
	logging.info("Plotting with plotly")
	#print(merged_sub.columns)

	
	custom_1 = merged_sub.filter(["ID", "Gene"])
	if args.cnv:
		template_1 = """<b>Rank:</b> %{x}<br><b>logPostOC:</b> %{y}<br><b>ID:</b> %{customdata[0]}<br>"""

	else:
		template_1 = """<b>Rank:</b> %{x}<br><b>logPostOC:</b> %{y}<br><b>ID:</b> %{customdata[0]}<br><b>Gene:</b> <i>%{customdata[1]}</i><br>"""

	custom_2 = merged_sub.filter(['logBF', 'AFF_CARR', 'AFF_NON-CARR', 'UNAFF_CARR', 'UNAFF_NON-CARR', 'MISS'])
	custom_2['AFF'] =  custom_2['AFF_CARR'] + custom_2['AFF_NON-CARR']
	custom_2['UNAFF'] =  custom_2['UNAFF_CARR'] + custom_2['UNAFF_NON-CARR']
	custom_2 = custom_2.filter(['logBF', 'AFF_CARR', 'AFF', 'UNAFF_CARR', 'UNAFF', 'MISS'])
	template_2 = """<b>logBF:</b> %{y}<br><b>AFF:</b> %{customdata[1]} / %{customdata[2]}<br><b>UNAFF:</b> %{customdata[3]} / %{customdata[4]}<br><b>MISS:</b> %{customdata[5]}<br>"""

	custom_3 = merged_sub.filter(["csq", "impact"] + keysPredictors).fillna('N/A')
	custom_3[args.frequency] = custom_3[args.frequency].round(6)

	if args.cnv:
		template_3 = """<b>logPriorOC:</b> %{y}<br>"""
		i = 0

	else:
		template_3 = """<b>logPriorOC:</b> %{y}<br><b>CSQ:</b> %{customdata[0]}<br><b>IMPACT:</b> %{customdata[1]}<br>"""
		i = 2

	for pred in keysPredictors:
		template_3 = template_3 + "<b>"  + pred + ":</b> %{customdata[" + str(i) + "]}<br>"
		i = i+1


	fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes='all', row_titles=["logPostOC", "logBF", "logPriorOC"], x_title="Rank")
	fig.append_trace(go.Bar(x=merged_sub["Rank"], y=merged_sub["logPostOC"], name='logPostOC', marker_color="#61D04F", customdata=custom_1, hovertemplate=template_1), 1, 1)
	fig.append_trace(go.Bar(x=merged_sub["Rank"], y=merged_sub["logBF"], name='logBF', marker_color="#2297E6", customdata=custom_2, hovertemplate=template_2), 2, 1)
	fig.append_trace(go.Bar(x=merged_sub["Rank"], y=merged_sub["logPriorOC"], name='logPriorOC', marker_color="#DF536B", customdata=custom_3, hovertemplate=template_3), 3, 1)
	fig.add_shape(go.layout.Shape(type="line", x0=0, y0=max_logBF, x1=args.top, y1=max_logBF, line=dict(dash="dash", width=3),
    ),row=2,col=1)
	fig.add_hline(y=max_logBF, line_dash="dash", row=2, col=1)
	fig.update_layout(showlegend=False, xaxis=dict(tickmode='array', tick0=1, ticktext=x_ticks_label, tickvals=x_ticks), plot_bgcolor='white')
	fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside', gridcolor='lightgrey')
	fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, ticks='outside')
	
	plotly.offline.plot(fig, filename=args.outputDir + args.prefix + ".BICEP.html")


	
	logging.info(" ")
	logging.info("Done")
	logging.info(" ")
	logging.info("--------------------------------------------------")
	logging.info(" ")




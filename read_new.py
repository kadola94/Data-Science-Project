import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import h5py
import pytz
import time
import datetime
from calendar import timegm

colors = ["red","orange","deepskyblue","black","lightgreen"]
years = [2014,2015,2016,2017,2018]
header = ["Day","Airline Code","Airline Name","Call Sign","Movement LSV","AC Type","Dest 3-Letter","Dest 4-Letter","ATA_ATD_ltc_Time","STA_STD_ltc_Time","RWY","ATM Def","Sitze"]
dataframe = pd.DataFrame(columns=header)

for i in range(len(years)):
	year = years[i]
	color = colors[i]
	filename = str(year) + ".txt"

	if i == 0:
		df = pd.read_csv(filename, skiprows = 1, encoding = "utf-16",delimiter ="\t", header = None, names = header)
		date_format = "%d.%m.%Y %H:%M:%S"
		days = df["Day"]
		times = df["ATA_ATD_ltc_Time"]
		times_planned = df["STA_STD_ltc_Time"]
		df['TIMESTAMP'] = pd.to_datetime(days + " " + times, format=date_format, errors="coerce")
		date_format = "%H:%M:%S"
		df['DELAY'] = (pd.to_datetime(times, format=date_format, errors="coerce") - pd.to_datetime(times_planned,
																								   format=date_format,
																								   errors="coerce"))
		# n,bins,patches = plt.hist(df["DELAY"].astype('timedelta64[ns]'),1,color=color,label=year)
		dataframe = dataframe.append(df, ignore_index = True)
	else:
		df = pd.read_csv(filename, skiprows = 1, encoding = "utf-16",delimiter ="\t", header = None, names = header)
		date_format = "%d.%m.%Y %H:%M:%S"
		days = df["Day"]
		times = df["ATA_ATD_ltc_Time"]
		times_planned = df["STA_STD_ltc_Time"]
		df['TIMESTAMP'] = pd.to_datetime(days + " " + times, format=date_format, errors="coerce")
		date_format = "%H:%M:%S"
		df['DELAY'] = (pd.to_datetime(times, format=date_format, errors="coerce") - pd.to_datetime(times_planned,
																								   format=date_format,
																								   errors="coerce"))
		# n,bins,patches = plt.hist(df["DELAY"].astype('timedelta64[ns]'),1,color=color,label=year)
		dataframe = dataframe.append(df, ignore_index = True)



def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)



def optimize(df):
	print(mem_usage(df))
	gl_obj = df.select_dtypes(include=['object']).copy()
	optimized_gl = df.copy()
	converted_obj = pd.DataFrame()

	for col in gl_obj.columns:
		num_unique_values = len(gl_obj[col].unique())
		num_total_values = len(gl_obj[col])
		if num_unique_values / num_total_values < 0.5:
			converted_obj.loc[:, col] = gl_obj[col].astype('category')
		else:
			converted_obj.loc[:, col] = gl_obj[col]

	print(mem_usage(gl_obj))
	print(mem_usage(converted_obj))

	compare_obj = pd.concat([gl_obj.dtypes, converted_obj.dtypes], axis=1)
	compare_obj.columns = ['before', 'after']
	compare_obj.apply(pd.Series.value_counts)
	optimized_gl[converted_obj.columns] = converted_obj
	mem_usage(optimized_gl)
	return optimized_gl

dataframe = optimize(dataframe)
print(dataframe.iloc[0])
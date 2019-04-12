import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import h5py
import pytz
import time
import datetime
import sys
from calendar import timegm
from collections import Counter


class Engine:

	def __init__(self,debug=False):
		self.db = debug
		self.years = [2014,2015,2016,2017,2018]
		self.df = None

	def print(self,string):
		sys.stdout.write(string)
		sys.stdout.flush()

	def load_dataset(self):
		header = ["Day","Airline Code","Airline Name","Call Sign","Movement LSV","AC Type","Dest 3-Letter","Dest 4-Letter","ATA_ATD_ltc_Time","STA_STD_ltc_Time","RWY","ATM Def","Sitze"]
		counter = 1
		for year in self.years:
			if self.db:
				self.print("[  ] loading dataset... {}/{}  [{}]\r".format(counter,len(self.years),"#"*counter*int(10/len(self.years)) + " "*(10-counter*int(10/len(self.years)))))
				counter += 1
			filename = str(year) + ".txt"
			if year > 2014:
				self.df = self.df.append(pd.read_csv(filename, skiprows = 1, encoding = "utf-16",delimiter ="\t",names=header))
			else:
				self.df = pd.read_csv(filename, skiprows = 1, encoding = "utf-16",delimiter ="\t",names=header)
		if self.db:
			self.print("[OK]\n")

	def get_timestamps(self):
		if self.db:
			self.print("[  ] loading timestamps...\r")
		date_format 	= "%d.%m.%Y %H:%M:%S"
		days 			= self.df["Day"]
		times 			= self.df["ATA_ATD_ltc_Time"]
		times_planned 	= self.df["STA_STD_ltc_Time"]
		self.df['TIMESTAMP'] 	= pd.to_datetime(days + " " + times,format=date_format,errors="coerce")
		self.df["YEAR"] 		= pd.to_datetime(days + " " + times,format=date_format,errors="coerce").dt.year
		self.df["WEEKDAYS"] 	= pd.to_datetime(days + " " + times,format=date_format,errors="coerce").dt.dayofweek

		if self.db:
			self.print("[OK] loading timestamps completed\n")

	def get_delay(self):
		if self.db:
			self.print("[  ] loading delay...\r")
		date_format = "%H:%M:%S"
		times 			= self.df["ATA_ATD_ltc_Time"]
		times_planned 	= self.df["STA_STD_ltc_Time"]
		self.df['DELAY'] = (pd.to_datetime(times,format=date_format,errors="coerce") - pd.to_datetime(times_planned,format=date_format,errors="coerce"))
		if self.db:
			self.print("[OK] loading delay completed \n")

	def get_wingspans(self,filename="wingspans.txt"):
		if self.db:
			self.print("[  ] loading wingspans...\r")
		header = ["AC TYPE","Wingspans"]
		wing_df = pd.read_csv(filename, encoding = "utf-8",delimiter ="\t",names=header)
		rename_dict = wing_df.set_index('AC TYPE').to_dict()['Wingspans']
		if self.db:
			self.print("[OK] loading wingspans completed \n")
		#doesnt work
		#self.df = self.df.replace(rename_dict)

	def optimize(self):
		gl_obj = self.df.select_dtypes(include=['object']).copy()
		optimized_gl = self.df.copy()
		converted_obj = pd.DataFrame()

		for col in gl_obj.columns:
			num_unique_values = len(gl_obj[col].unique())
			num_total_values = len(gl_obj[col])
			if num_unique_values / num_total_values < 0.5:
				converted_obj.loc[:, col] = gl_obj[col].astype('category')
			else:
				converted_obj.loc[:, col] = gl_obj[col]

		compare_obj = pd.concat([gl_obj.dtypes, converted_obj.dtypes], axis=1)
		compare_obj.columns = ['before', 'after']
		compare_obj.apply(pd.Series.value_counts)
		optimized_gl[converted_obj.columns] = converted_obj
		self.df = optimized_gl

	def mem_usage(pandas_obj):
		if isinstance(pandas_obj, pd.DataFrame):
			usage_b = pandas_obj.memory_usage(deep=True).sum()
		else:  # we assume if not a df it's a series
			usage_b = pandas_obj.memory_usage(deep=True)
		usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
		return "{:03.2f} MB".format(usage_mb)


if __name__ == "__main__":

	Airport = Engine(True)
	Airport.load_dataset()
	Airport.get_timestamps()
	Airport.get_delay()
	Airport.get_wingspans()
	Airport.optimize()
	# Airport.write_to_csv

	# AC_counts = Counter(Airport.df["AC Type"])
	# df = pd.DataFrame.from_dict(AC_counts, orient='index')
	# df.plot(kind='bar')
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(df)








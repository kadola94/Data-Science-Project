#from mpl_toolkits.basemap import Basemap
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

	def write(self,string):
		sys.stdout.write(string)
		sys.stdout.flush()

	def load_airport(self):
		header = ["Day","Airline Code","Airline Name","Call Sign","Movement LSV","AC Type","IATA","ICAO","ACTUAL","EXPECTED","RWY","ATM Def","Sitze"]
		counter = 1
		for year in self.years:
			if self.db:
				self.write("[  ] loading dataset... {}/{}  [{}]\r".format(counter,len(self.years),"#"*counter*int(10/len(self.years)) + " "*(10-counter*int(10/len(self.years)))))
				counter += 1
			filename = str(year) + ".txt"
			if year > 2014:
				self.df = self.df.append(pd.read_csv(filename, skiprows = 1, encoding = "utf-16",delimiter ="\t",names=header))
			else:
				self.df = pd.read_csv(filename, skiprows = 1, encoding = "utf-16",delimiter ="\t",names=header)
		if self.db:
			self.write("[OK]\n")

	def get_timestamps(self):
		if self.db:
			self.write("[  ] loading timestamps...\r")
		date_format 	= "%d.%m.%Y %H:%M:%S"
		days 			= self.df["Day"]
		times 			= self.df["ACTUAL"]
		times_planned 	= self.df["EXPECTED"]
		self.df['TIMESTAMP'] 	= pd.to_datetime(days + " " + times,format=date_format,errors="coerce")
		self.df["YEAR"] 		= pd.to_datetime(days + " " + times,format=date_format,errors="coerce").dt.year
		self.df["WEEKDAYS"] 	= pd.to_datetime(days + " " + times,format=date_format,errors="coerce").dt.dayofweek
		if self.db:
			self.write("[OK] loading timestamps completed\n")

	def get_delay(self):
		if self.db:
			self.write("[  ] loading delay...\r")
		date_format = "%H:%M:%S"
		times 			= self.df["ACTUAL"]
		times_planned 	= self.df["EXPECTED"]
		self.df['DELAY'] = (pd.to_datetime(times,format=date_format,errors="coerce") - pd.to_datetime(times_planned,format=date_format,errors="coerce"))
		if self.db:
			self.write("[OK] loading delay completed \n")

	def get_wingspans(self,filename="wingspans.csv"):
		if self.db:
			self.write("[  ] loading wingspans...\r")
		header = ["AC Type","Wingspans"]
		wing_df = pd.read_csv(filename, encoding = "utf-8",delimiter ="\t",names=header)
		if self.db:
			self.write("[OK] loading wingspans completed \n")
		#wing_df.set_index('AC TYPE')
		#self.df.set_index("AC TYPE",inplace=True)
		self.df = pd.merge(self.df, wing_df, on="AC Type")

	def get_coordinates(self,filename="airports.csv"):
		header = ["name","city","country","IATA","ICAO","lat","long","altitude","timezone","DST"]
		codes_df = pd.read_csv(filename, encoding = "utf-8", skiprows=0,delimiter =",",names=header)
		codes_df.set_index("IATA",inplace=True)
		codes_df = codes_df.loc[:,"lat":"long"]
		print(codes_df.loc["ZRH",:])
		#self.df.set_index("IATA",inplace=True)
		self.df = pd.merge(self.df, codes_df, on="IATA")
		self.plot_routes()

	def plot_routes(self):
		ZRHx = 47.464722
		ZRHy = 8.549167
		route_counts = Counter(self.df["IATA"])
		route_counts = {"IATA":route_counts.keys(),"counts":route_counts.values()}
		routes = pd.DataFrame.from_dict(route_counts)
		self.df.set_index("IATA",inplace=True)
		coords = self.df.loc[:,"lat":"long"]
		routes = pd.merge(routes,coords,on="IATA")
		routes.drop_duplicates(keep="first", inplace=True)

		m = Basemap()
		m.drawcoastlines()
		x = np.array(routes.loc[:,"lat"])
		y = np.array(routes.loc[:,"long"])
		alphas = np.array(routes.loc[:,"counts"])
		n = max(alphas)/100

		for xi,yi,a in zip(x,y,alphas):
			a = max(min(a/n,1),0.2)	#set alpha depending on counts
			m.drawgreatcircle(ZRHy,ZRHx,yi,xi,alpha=a,linewidth=1,color="deepskyblue")
		plt.tight_layout()
		plt.savefig("flights.png",dpi=300)
		plt.show()

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

	def mem_usage(self,pandas_obj):
		if isinstance(pandas_obj, pd.DataFrame):
			usage_b = pandas_obj.memory_usage(deep=True).sum()
		else:  # we assume if not a df it's a series
			usage_b = pandas_obj.memory_usage(deep=True)
		usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
		return "{:03.2f} MB".format(usage_mb)


class Weather:

	def __init__(self,debug=False):
		self.db = debug
		self.df = None

	def write(self,string):
		sys.stdout.write(string)
		sys.stdout.flush()

	def load_waether(self,filename = "weather.csv"):
		header = ["Wetterstation","Time","Boeenspitze 3s","Boeenspitze 1s","Temp","Niederschlag","Visibility","Windgeschwindigkeit","Windrichtung"]
		if self.db:
			self.write("[  ] loading dataset...\r")
		self.weather = pd.read_csv(filename, skiprows = 1, encoding = "utf-8",delimiter =",",names=header)
		if self.db:
			self.write("[OK]\n")

	def get_times(self):
		if self.db:
			self.write("[  ] loading times...\r")
		date_format 	= "%Y%m%d%H%M"
		days 			= self.weather["Time"]
		self.weather['Time'] = pd.to_datetime(days, format=date_format, errors="coerce")
		print(self.weather['Time'])
		if self.db:
			self.write("[OK] loading times completed\n")

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

	def mem_usage(self,pandas_obj):
		if isinstance(pandas_obj, pd.DataFrame):
			usage_b = pandas_obj.memory_usage(deep=True).sum()
		else:  # we assume if not a df it's a series
			usage_b = pandas_obj.memory_usage(deep=True)
		usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
		return "{:03.2f} MB".format(usage_mb)

if __name__ == "__main__":

	# Airport = Engine(True)
	# Airport.load_dataset()
	# Airport.get_timestamps()
	# Airport.get_delay()
	# Airport.get_wingspans()
	# Airport.get_coordinates()
	# Airport.optimize()

	Weather = Weather(True)
	Weather.load_waether()
	Weather.get_times()



	#plt.show()
	#AC_counts = Counter(Airport.df["AC Type"])
	#df = pd.DataFrame.from_dict(AC_counts, orient='index')
	#df.plot(kind='bar')
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(Airport.df.head())








# from mpl_toolkits.basemap import Basemap
# note to linus: works only in python 2.7
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

	def load_data(self,filename="dataset.txt"):
		header = ["Timestamp","Date","Year","Weekday",\
		"T expected","T actual","Delay","Airline Code",\
		"Airline Name","Call Sign","Movement LSV","RWY",\
		"ATM Def","AC Type","Wingspans","Seats","IATA","ICAO",\
		"Destination Lat","Destination Long","Gusts 1s","Gusts 3s",\
		"Temperature","Precipitation","Visibility","Wind Speed","Wind Direction"]
		if self.db:
			self.write("[  ] loading dataset... \r")
		self.df = pd.read_csv(filename, skiprows = 1, encoding = "utf-8",delimiter =",",names=header)
		if self.db:
			self.write("[OK]\n")

	def plot_routes(self):
		ZRHx = 47.464722
		ZRHy = 8.549167
		route_counts = Counter(self.df["IATA"])
		route_counts = {"IATA":list(route_counts.keys()),"counts":list(route_counts.values())}
		routes = pd.DataFrame.from_dict(route_counts)
		self.df.set_index("IATA",inplace=True)
		coords = self.df.loc[:,"Destination Lat":"Destination Long"]
		routes = pd.merge(routes,coords,on="IATA")
		routes.drop_duplicates(keep="first", inplace=True)

		m = Basemap()
		m.drawcoastlines()
		x = np.array(routes.loc[:,"Destination Lat"])
		y = np.array(routes.loc[:,"Destination Long"])
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

if __name__ == "__main__":

	Airport = Engine(True)
	Airport.load_data()
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(Airport.df.head())
	#Airport.plot_routes()

	#plt.show()
	#AC_counts = Counter(Airport.df["AC Type"])
	#df = pd.DataFrame.from_dict(AC_counts, orient='index')
	#df.plot(kind='bar')
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(Airport.df.head())








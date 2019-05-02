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

	def roundplots(self):
		rwys_n = [10, 14, 16, 28, 32, 34]
		windspeed_median = self.df['Wind Speed'].median()
		rwys = []
		for i in rwys_n:
			rwys.append(self.df.loc[(self.df["RWY"] == i) & (self.df["Wingspans"] < 60) & (self.df["Wind Speed"] > windspeed_median)])
		fig = plt.figure()
		fig.set_size_inches(19.2, 10.8)
		bin_size = 10
		for i in range(len(rwys_n)):
			runway = rwys[i]
			a, b = np.histogram(runway['Wind Direction'], bins = np.arange(0, 360 + bin_size, bin_size))
			centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])
			i = 231+ i
			ax = fig.add_subplot(i, projection = 'polar')
			ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
			ax.set_theta_zero_location("N")
			ax.set_theta_direction(-1)
			figure_title = 'Runway {}'.format(rwys_n[i-231])
			plt.text(0.5, 1.15, figure_title, horizontalalignment='center', fontsize=15, transform = ax.transAxes)
			#ax.set_title('Runway {}'.format(rwys_n[i-331]), verticalalignment = 'bottom')
			# line = zip(np.deg2rad(rwys_n[i-331]*10), np.max(a))
			ax.plot([np.deg2rad(rwys_n[i-231]*10), np.deg2rad(rwys_n[i-231]*10)], [0, np.max(a)], c = 'r')
			# ax.plot((0, line[0]), (0, line[1]), c = 'r',zorder = 3)

		fig.tight_layout(pad=1, h_pad=2.0)
		plt.savefig('ohne_grosse.pdf', dpi = 100)
		plt.show()

if __name__ == "__main__":

	RADAR = Engine(True)
	RADAR.load_data()
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	#	print(RADAR.df.head())

	RADAR.roundplots()




	#RADAR.plot_routes()
	#
	# wind_speed = RADAR.df["Wind Direction"]
	# wingspans = RADAR.df["Wingspans"]
	# RADAR.df['Wind Direction'].hist(by=RADAR.df['RWY'], range=(0, RADAR.df['Wind Direction'].max()), bins = 36, density = True)
	# print(max(wind_speed))
	# plt.scatter(wind_speed,wingspans)
	# plt.show()
	# AC_counts = Counter(RADAR.df["AC Type"])
	# df = pd.DataFrame.from_dict(AC_counts, orient='index')
	# df.plot(kind='bar')
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(RADAR.df.head())








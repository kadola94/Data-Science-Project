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
import datetime as dt
import sys
from calendar import timegm
from collections import Counter
import matplotlib.dates as md


class Engine:

	def __init__(self,debug=False):
		self.db = debug
		self.years = [2014,2015,2016,2017,2018]
		self.df = None
		self.RWYS = ['10','14','16','28','32','34']
		self.n_days = 1


	def write(self,string):
		sys.stdout.write(string)
		sys.stdout.flush()

	def load_data(self,filename="dataset.txt"):
		header  = ["Timestamp","UNIX","Date","Year","Weekday","Hour",\
		"T expected","T expected UT","T actual","T actual UT","Delay","Airline Code",\
		"Airline Name","Call Sign","Movement LSV","RWY",\
		"ATM Def","AC Type","Wingspans","Seats","IATA","ICAO",\
		"Destination Lat","Destination Long","Gusts 1s","Gusts 3s",\
		"Temperature","Precipitation","Visibility","Wind Speed","Wind Direction"]
		if self.db:
			self.write("[  ] loading dataset... \r")
		self.df = pd.read_csv(filename, skiprows = 1, encoding = "utf-8",delimiter =",",names=header, parse_dates= ["Timestamp"])
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

	def get_average(self):
		avg = {}
		for RWY in self.RWYS:
			tmp = {}
			for month in range(1,13):
				tmp[month] = []
			avg[RWY] = {
						"weekdays" : tmp,
						"weekends" : tmp
			}

		self.n_days = 	len(Counter(RADAR.df["Date"]))
		print("days in dataset: {}".format(self.n_days))

		for RWY in self.RWYS:
			for month in range(1,13):
				weekdays = RADAR.df[RADAR.df["Weekday"].astype(np.int) < 5]
				weekends = RADAR.df[RADAR.df["Weekday"].astype(np.int) >= 5]

				weekdays = weekdays[weekdays["Timestamp"].dt.month.astype(np.int) == month]
				weekends = weekends[weekends["Timestamp"].dt.month.astype(np.int) == month]

				weekdays = weekdays[ weekdays["RWY"].astype(np.int) == int(RWY)]
				weekends = weekends[ weekends["RWY"].astype(np.int) == int(RWY)]

				wd = []
				we = []
				for hour in range(0,24):
					#wd.append(len(weekdays[(weekdays["T actual UT"] >= 946684800*1000 + hour*60*60*1000) &(weekdays["T actual UT"] <= 946684800*1000 + (hour+1)*60*60*1000)])/(self.n_days*5/7/24))
					#we.append(len(weekends[(weekends["T actual UT"] >= 946684800*1000 + hour*60*60*1000) &(weekends["T actual UT"] <= 946684800*1000 + (hour+1)*60*60*1000)])/(self.n_days*2/7/24))

					wd.append(len(weekdays[weekdays["Hour"] == hour])/(self.n_days*5/7/24))
					we.append(len(weekends[weekends["Hour"] == hour])/(self.n_days*2/7/24))


				#print(wd)
				avg[RWY]["weekdays"][month] = wd
				avg[RWY]["weekends"][month] = we
		return avg





	# def optimize(self):
	# 	gl_obj = self.df.select_dtypes(include=['object']).copy()
	# 	optimized_gl = self.df.copy()
	# 	converted_obj = pd.DataFrame()

	# 	for col in gl_obj.columns:
	# 		num_unique_values = len(gl_obj[col].unique())
	# 		num_total_values = len(gl_obj[col])
	# 		if num_unique_values / num_total_values < 0.5:
	# 			converted_obj.loc[:, col] = gl_obj[col].astype('category')
	# 		else:
	# 			converted_obj.loc[:, col] = gl_obj[col]

	# 	compare_obj = pd.concat([gl_obj.dtypes, converted_obj.dtypes], axis=1)
	# 	compare_obj.columns = ['before', 'after']
	# 	compare_obj.apply(pd.Series.value_counts)
	# 	optimized_gl[converted_obj.columns] = converted_obj
	# 	self.df = optimized_gl

	# def mem_usage(self,pandas_obj):
	# 	if isinstance(pandas_obj, pd.DataFrame):
	# 		usage_b = pandas_obj.memory_usage(deep=True).sum()
	# 	else:  # we assume if not a df it's a series
	# 		usage_b = pandas_obj.memory_usage(deep=True)
	# 	usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
	# 	return "{:03.2f} MB".format(usage_mb)

if __name__ == "__main__":

	RADAR = Engine(True)
	RADAR.load_data()
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(RADAR.df.head())
	#RADAR.plot_routes()		
	#RADAR.df.dropna(how='any',inplace = True)
	avg = RADAR.get_average()
	RWYS = ['10','14','16','28','32','34']
	months = range(1,13)
	for month in months:
		fig, axes = plt.subplots(2,int(len(RWYS)/2),figsize=(18,6),sharey=True)
		axes = axes.reshape(1,len(RWYS))[0]
		for RWY,ax in zip(RWYS,axes):
			weekdays = RADAR.df[RADAR.df["Weekday"].astype(np.int) < 5]
			weekends = RADAR.df[RADAR.df["Weekday"].astype(np.int) >= 5]

			weekdays = weekdays[weekdays["Timestamp"].dt.month.astype(np.int) == month]
			weekends = weekends[weekends["Timestamp"].dt.month.astype(np.int) == month]
			ax.set_title(RWY)
			#y = RADAR.df["Delay"]/np.timedelta64(1,'s')

			weekdays = weekdays[ weekdays["RWY"].astype(np.int) == int(RWY)]
			weekends = weekends[ weekends["RWY"].astype(np.int) == int(RWY)]

			#print(weekdays)
			#width = (a_bins[1] - a_bins[0])/3

			#print(a_bins)
			#print(b_bins)

			xfmt = md.DateFormatter('%H:%M')
			ax.xaxis.set_major_formatter(xfmt)
			ax.set_xlim(dt.datetime(2000,1,1,0,0,0,0),dt.datetime(2000,1,1,23,59,59,0))
			norm = False
			if norm:
				#a_heights, a_bins = np.histogram(weekdays["T actual UT"], bins= 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) )
				#b_heights, b_bins = np.histogram(weekends["T actual UT"], bins= a_bins)
				a_heights, a_bins = np.histogram(weekdays["Hour"], bins= np.arange(0,25) )
				b_heights, b_bins = np.histogram(weekends["Hour"], bins= a_bins)
				#print(a_bins)

				a_bins = 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) - 90*1000
				b_bins = 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) - 90*1000
				a_bins =[dt.datetime.fromtimestamp(ts/1000) for ts in a_bins]
				b_bins =[dt.datetime.fromtimestamp(ts/1000) for ts in b_bins]
				norm_we = (RADAR.n_days*2/7/24)
				norm_wd = (RADAR.n_days*5/7/24)

				#ax.bar(np.array(a_bins[:-1])-np.array([dt.timedelta(minutes=30) for i in a_bins[:-1]]), a_heights/5, width = 0.5 * (1 / 24), facecolor='cornflowerblue')
				ax.bar(a_bins, b_heights/norm_we, width = (1 / 24), alpha=0.5, facecolor='deepskyblue',	edgecolor="black")
				ax.bar(b_bins, a_heights/norm_wd, width = (1 / 24), alpha=0.5, facecolor='cornflowerblue',edgecolor="black")
			else:
				num_day = 0
				weekdays = weekdays[ weekdays["Date"] == '11.{:02d}.2016'.format(month)]
				weekends = weekends[ weekends["Date"] == '11.{:02d}.2016'.format(month)]
				#a_heights, a_bins = np.histogram(weekdays["T actual UT"], bins= 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) )
				#b_heights, b_bins = np.histogram(weekends["T actual UT"], bins= a_bins)
				a_heights, a_bins = np.histogram(weekdays["Hour"], bins= np.arange(0,25) )
				b_heights, b_bins = np.histogram(weekends["Hour"], bins= a_bins)
				#print(a_bins)
				a_bins = 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) - 90*1000
				b_bins = 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) - 90*1000
				a_bins =[dt.datetime.fromtimestamp(ts/1000) for ts in a_bins]
				b_bins =[dt.datetime.fromtimestamp(ts/1000) for ts in b_bins]

				#ax.bar(np.array(a_bins[:-1])-np.array([dt.timedelta(minutes=30) for i in a_bins[:-1]]), a_heights/5, width = 0.5 * (1 / 24), facecolor='cornflowerblue')
				ax.bar(a_bins, b_heights, width = (1 / 24), alpha=0.5, facecolor='deepskyblue',	edgecolor="black")
				ax.bar(b_bins, a_heights, width = (1 / 24), alpha=0.5, facecolor='cornflowerblue',edgecolor="black")			
			for hour in range(0,24):
				ax.scatter(dt.datetime(2000,1,1,hour,0,0,0),avg[RWY]["weekdays"][month][hour],color="red")
			#ax.set_aspect('auto')


		'''
		counts_wd = Counter(weekdays["T actual UT"])
		counts_we = Counter(weekends["T actual UT"])
		counts_wd = pd.DataFrame.from_dict(counts_wd, orient='index')
		counts_we = pd.DataFrame.from_dict(counts_we, orient='index')
		print(counts_we)
		# counts_wd.plot.hist(bins=1000)
		# counts_we.plot.hist(bins=1000)
		fig, ax  = plt.subplots()
		ax.bar(counts_we.index.values, counts_we, facecolor='cornflowerblue')
		ax.bar(counts_wd.index.values, counts_wd, facecolor='seagreen')
		'''
		#plt.xlim(0,40)
		plt.tight_layout()
		plt.savefig(str(month)+".png",dpi=250)
		print("saved month={}".format(month))
		plt.cla()
		plt.gca()
		# print(counts.head())
		# #y = y[:bound]
		# #y = y[x.index.values]
		# x = range(5)
		# y = counts
		# print(x)
		# print(y)
		# ax.scatter(x,y)
		# plt.show()
		#AC_counts = Counter(RADAR.df["AC Type"])
		#df = pd.DataFrame.from_dict(AC_counts, orient='index')
		#df.plot(kind='bar')
		# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		# 	print(RADAR.df.head())

	# performance calculation takes ages
	exit()

	performance = []
	fig, axes = plt.subplots(2,int(len(RWYS)/2),figsize=(18,6),sharey=True)
	axes = axes.reshape(1,len(RWYS))[0]
	for RWY,ax in zip(RWYS,axes):
		for year in RADAR.years:
			for month in range(1,13):
				print("year: {}, month: {}".format(year,month))
				for day in range(1,32):
					for hour in range(0,23):
						weekdays = RADAR.df[RADAR.df["Weekday"].astype(np.int) < 5]

						weekdays = weekdays[ weekdays["RWY"].astype(np.int) == int(RWY)]

						weekdays = weekdays[ weekdays["Date"] == '{:02d}.{:02d}.{}'.format(day,month,year)]

						#a_heights, a_bins = np.histogram(weekdays["T actual UT"], bins= 946684800*1000+np.arange(0,24*60*60*1000,60*60*1000) )
						a_heights, a_bins = np.histogram(weekdays["Hour"], bins= np.arange(0,24) )

						error = avg[RWY]["weekdays"][month][hour] - a_heights[hour]
						print(error)
						performance.append( error )
		ax.plot(range(len(performance)),performance)
	plt.tight_layout()
	plt.show()







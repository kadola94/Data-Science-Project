# from mpl_toolkits.basemap import Basemap
# note to linus: works only in python 2.7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.optimize import curve_fit
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

	def wind_directions(self):
		rwys_n = [10, 14, 16, 28, 32, 34]
		windspeed_average = self.df['Wind Speed'].mean()
		rwys = []
		for i in rwys_n:
			rwys.append(self.df.loc[(self.df["RWY"] == i) & (self.df["Wind Speed"] > 2) & (self.df["Movement LSV"] == 'L') & (self.df["Wind Direction"] <= 360) & (self.df["Wind Direction"] >= 180)])
		fig = plt.figure()
		fig.set_size_inches(19.2, 10.8)
		bin_size = 10
		for i in range(len(rwys_n)):
			runway = rwys[i]
			a, b = np.histogram(runway['Wind Direction'], bins = np.arange(0, 360 + bin_size, bin_size))
			centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])
			ax = fig.add_subplot(231 + i, projection = 'polar')
			ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
			ax.set_theta_zero_location("N")
			ax.set_theta_direction(-1)
			figure_title = 'Runway {}'.format(rwys_n[i])
			plt.text(0.5, 1.15, figure_title, horizontalalignment='center', fontsize=15, transform = ax.transAxes)
			#ax.set_title('Runway {}'.format(rwys_n[i-331]), verticalalignment = 'bottom')
			# line = zip(np.deg2rad(rwys_n[i-331]*10), np.max(a))
			ax.plot([np.deg2rad(rwys_n[i]*10), np.deg2rad(rwys_n[i]*10)], [0, np.max(a)], c = 'r')
			# ax.plot((0, line[0]), (0, line[1]), c = 'r',zorder = 3)

		fig.tight_layout(pad=1, h_pad=2.0)
		plt.show()
		plt.savefig('find_wind_threshold.pdf', dpi = 100)

	def gausss(self, x, sigma, mu, offset, amplitude):
		amplitude = np.linalg.norm(amplitude)
		return amplitude * np.exp(-(x-mu)**2/sigma**2) + offset

	def wind_threshold(self):
		rwys_n = [10, 14, 16, 28, 32, 34]
		rwys = []
		for i in rwys_n:
			rwys.append(self.df.loc[(self.df["RWY"] == i) & (self.df["Wind Speed"] > 0) & (self.df["Movement LSV"] == 'L') & (self.df["Wind Direction"] <= 360) & (self.df["Wind Direction"] >= 0)])
		fig = plt.figure()
		fig.set_size_inches(19.2, 10.8)
		n_bins = 150
		bin_x = np.linspace(0,360, n_bins)
		bin_y = np.linspace(0, 20, n_bins)
		thresholds = []
		for i in range(len(rwys_n)):
			plt.subplot(2,3,i+1)
			runway = rwys[i]
			x = np.array(runway['Wind Direction'])
			y = np.array(runway['Wind Speed'])
			title = 'Runway {}'.format(rwys_n[i])
			plt.title(title)
			plt.xlabel('Wind Direction [°]')
			plt.ylabel('Wind Speed [m/s]')
			plt.hist2d(x,y, bins = (bin_x, bin_y), cmin = 1)
			plt.colorbar()
			# ind = np.where(y == max(y))[0][0]
			# mu = x[ind]
			# amplitud = max(y)
			# p0 = [30, mu, 2, amplitud]
			# popt, pcov = curve_fit(self.gausss, x, y, p0 = p0)
			# thresholds.append(popt[-2])
			# x = np.linspace(min(x),max(x),1000)
			# print('Threshold for Runway {} is {}'.format(rwys_n[i], popt[-2]))
			# plt.plot(x, self.gausss(x, *popt), c = 'r')
		fig.tight_layout(pad=1, h_pad=2.0)
		plt.savefig('wind_threshold_nofit.pdf', dpi=100)

	def vis_precip_correlations(self):
		visibility_limit = 10000
		precipitation_min = 1
		precipitation_max = self.df['Precipitation'].max()
		bins = 250
		binx_1 = np.linspace(0, visibility_limit, bins)
		biny = np.linspace(precipitation_min, precipitation_max, bins)

		x = visibility_limit - self.df['Visibility'].loc[(self.df['Precipitation'] > precipitation_min) & (self.df['Visibility'] < visibility_limit)]
		y = self.df['Precipitation'].loc[(self.df['Precipitation'] > precipitation_min) & (self.df['Visibility'] < visibility_limit)]
		a, b = np.polyfit(x, y, 1)
		x_lin = np.linspace(0,visibility_limit, 100)
		fig = plt.figure()
		fig.set_size_inches(10, 10)
		plt.title('Visibility - Precipitation')
		plt.xlabel('Anti - Visibility [m]')
		plt.ylabel('Precipitation [mm]')
		plt.hist2d(x,y, bins = (binx_1, biny), cmin = 1, label = 'Data Points')
		plt.colorbar()
		plt.plot(x_lin, x_lin*a + b, '-r', label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a, b))
		plt.legend(frameon=True, title='Visibility < {}km, Precipitation > {}mm'.format(visibility_limit/1000, precipitation_min))
		plt.savefig('Visibilty_vs_Precipitation.pdf', dpi = 100)

	def wind_correlations(self):
		gust_max_1 = self.df['Gusts 1s'].max()
		gust_max_2 = self.df['Gusts 3s'].max()
		wind_speed_max = self.df['Wind Speed'].max()

		bins = 250
		binx_1 = np.linspace(0, gust_max_1, bins)
		binx_2 = np.linspace(0, gust_max_2, bins)
		biny = np.linspace(0, wind_speed_max, bins)
		x_1 = self.df['Gusts 1s'].loc[(self.df['Gusts 1s'] >= 0) & (self.df['Wind Speed'] >= 0)]
		x_2 = self.df['Gusts 3s'].loc[(self.df['Gusts 3s'] >= 0) & (self.df['Wind Speed'] >= 0)]

		y = self.df['Wind Speed'].loc[(self.df['Gusts 3s'] >= 0) & (self.df['Wind Speed'] >= 0)]
		a_1, b_1 = np.polyfit(x_1, y, 1)
		a_2, b_2 = np.polyfit(x_2, y, 1)

		x_lin_1 = np.linspace(0, gust_max_1, 100)
		x_lin_2 = np.linspace(0, gust_max_2, 100)
		fig = plt.figure()
		fig.set_size_inches(12, 12)
		plt.subplots_adjust(wspace=0.5, hspace=0.5)


		plt.subplot(2,1,1)
		plt.title('Gusts 1s - Wind Speed')
		plt.xlabel('Gusts 1s [m/s]')
		plt.ylabel('Wind Speed [m/s]')
		plt.hist2d(x_1,y, bins = (binx_1, biny), cmin = 1, label = 'Data Points')
		plt.plot(x_lin_1, x_lin_1*a_1 + b_1, '-r', lw = 0.5, label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a_1, b_1))
		plt.colorbar()
		plt.legend(frameon=True)

		plt.subplot(2,1,2)
		plt.title('Gusts 3s - Wind Speed')
		plt.xlabel('Gusts 3s [m/s]')
		plt.ylabel('Wind Speed [m/s]')
		plt.hist2d(x_2,y, bins = (binx_2, biny), cmin = 1, label = 'Data Points')
		plt.plot(x_lin_2, x_lin_2*a_2 + b_2, '-r', lw = 0.5, label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a_2, b_2))
		plt.colorbar()
		plt.legend(frameon=True)

		fig.tight_layout(pad=1, h_pad=2.0)

		plt.savefig('Gusts_vs_Wind_Speed.pdf', dpi = 100)


	def delay_correlations(self):
		delay = pd.Series(pd.to_timedelta(self.df['Delay'], unit = 's'))
		anti_visibility = 50000 - self.df['Visibility']
		wind_speed = self.df['Wind Speed']
		precipication = self.df['Precipitation']
		temperature = self.df['Temperature']
		delay = delay.dt.total_seconds()/3600

		#delay_max = delay.max()
		#delay_min = delay.min()
		visibility_max = self.df['Visibility'].max()
		wind_speed_max = self.df['Wind Speed'].max()
		precipication_max = self.df['Precipitation'].max()
		T_max = self.df['Temperature'].max()
		T_min = self.df['Temperature'].min()

		bins = 250
		binx_visibility = np.linspace(0, visibility_max, bins)
		binx_wind_speed = np.linspace(0, wind_speed_max, bins)
		binx_precipitation = np.linspace(0, precipication_max, bins)
		binx_temperature = np.linspace(T_min, T_max, bins)

		biny = np.linspace(-5, 10, bins)


		# a_1, b_1 = np.polyfit(x_1, y, 1)
		# a_2, b_2 = np.polyfit(x_2, y, 1)

		# x_lin_1 = np.linspace(0, gust_max_1, 100)
		# x_lin_2 = np.linspace(0, gust_max_2, 100)
		fig = plt.figure()
		fig.set_size_inches(12, 12)
		plt.subplots_adjust(wspace=0.5, hspace=0.5)


		plt.subplot(2,2,1)
		plt.title('Delay - Visibility')
		plt.xlabel('Visibility [m]')
		plt.ylabel('Delay [h]')
		plt.hist2d(anti_visibility,delay, bins = (binx_visibility, biny), cmin = 1)
		# plt.plot(x_lin_1, x_lin_1*a_1 + b_1, '-r', lw = 0.5, label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a_1, b_1))
		plt.colorbar()

		plt.subplot(2,2,2)
		plt.title('Delay - Wind Speed')
		plt.xlabel('Wind Speed [m/s]')
		plt.ylabel('Delay [h]')
		plt.hist2d(wind_speed,delay, bins = (binx_wind_speed, biny), cmin = 1)
		# plt.plot(x_lin_2, x_lin_2*a_2 + b_2, '-r', lw = 0.5, label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a_2, b_2))
		plt.colorbar()

		plt.subplot(2,2,3)
		plt.title('Delay - Precipitation')
		plt.xlabel('Precipitation [mm]')
		plt.ylabel('Delay [h]')
		plt.hist2d(precipication,delay, bins = (binx_precipitation, biny), cmin = 1)
		# plt.plot(x_lin_2, x_lin_2*a_2 + b_2, '-r', lw = 0.5, label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a_2, b_2))
		plt.colorbar()

		plt.subplot(2,2,4)
		plt.title('Delay - Temperature')
		plt.xlabel('Temperature [°C]')
		plt.ylabel('Delay [h]')
		plt.hist2d(temperature,delay, bins = (binx_temperature, biny), cmin = 1)
		# plt.plot(x_lin_2, x_lin_2*a_2 + b_2, '-r', lw = 0.5, label =r'Fit: f(x) = a$\cdot$x + b with a = {:.2E} and b = {:.2E}'.format(a_2, b_2))
		plt.colorbar()
		#plt.legend(frameon=True)

		fig.tight_layout(pad=1, h_pad=2.0)

		plt.savefig('Delay_vs_everything.pdf', dpi = 100)

	def stats(self):
		print('Number of entries: {}'.format(self.df.shape[1]))
		movements = []
		years = [2014, 2015, 2016, 2017, 2018]
		for year in years:
			#percentage = self.df.loc[self.df['Year'] == year].shape[0]/self.df.loc[self.df['Year'] == year-1].shape[0]-1
			#print('Average in {}: {} \nIncrease: \t\t {:.2%}'.format(year, self.df.loc[self.df['Year'] == year].shape[0], percentage))
			movements.append(self.df.loc[self.df['Year'] == year].shape[0])
		mov_max, mov_min = max(movements), min(movements)
		movements = np.array(movements)

		fig = plt.figure()
		fig.set_size_inches(10, 10)
		plt.bar(years, movements)
		plt.axis([2013.5, 2018.5, mov_min-5000, mov_max + 1000])
		plt.xticks(years)
		plt.xlabel('Years')
		plt.ylabel('Flight Movements')
		plt.title('Flight Movement Trend')
		plt.savefig('Flight_Movement_Trend.pdf', dpi = 100)
		percentage = self.df.loc[self.df['Year'] == 2018].shape[0] / self.df.loc[self.df['Year'] == 2014].shape[0] - 1
		print('Total increase in Flight movements in the last 5 years: {:.2%}'.format(percentage))
		print('Max Wind Gusts: {:.2f}km/h'.format(self.df['Gusts 1s'].max() * 3.6))
		print('Min Visibility: {:.2f}m'.format(self.df['Visibility'].min()))

		# delay = pd.Series(pd.to_timedelta(self.df['Delay'], unit = 's'))
		# delay = delay.dt.total_seconds()
		# delay_max = delay.max()
		# print(plt.hist
	# 'Max delay: {:.2f} hours'.format(delay_max/3600))

	def avro(self):
		ac_types = ['RJ1H', 'BCS1', 'BCS3']
		swiss = 'Swiss International Air Lines AG'
		# delays:
		delays = []
		for type in ac_types:
			delay = pd.Series(pd.to_timedelta(self.df['Delay'].loc[(self.df['AC Type'] == type) & (self.df["Airline Name"] == swiss)], unit='s'))
			delay = np.array(delay.dt.total_seconds() / 60)
			mean = np.mean(delay)
			std = np.std(delay)
			delays.append(delay)
			print('for aircraft type {} we have: {:.2f} +/- {:.2f}'.format(type, mean, std))

		fig = plt.figure()
		fig.set_size_inches(10,10)
		plt.ylim(-40, 80)
		plt.ylabel('Delay [min]')
		plt.xlabel('Aircraft Types')
		plt.boxplot(delays)
		plt.xticks([1, 2, 3], ac_types)
		plt.savefig('jumbolino_delays.pdf', dpi = 100)
		plt.show()











if __name__ == "__main__":

	RADAR = Engine(True)
	RADAR.load_data()
	# RADAR.stats()
	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	#  	print(RADAR.df.head())

	# RADAR.wind_directions()
	# RADAR.vis_precip_correlations()
	# RADAR.wind_correlations()
	# RADAR.delay_correlations()
	# RADAR.wind_threshold()
	# RADAR.avro()







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








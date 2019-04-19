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
import scipy.interpolate as interp


class Engine:

	def __init__(self,debug=False):
		self.db = debug
		self.years = [2014,2015,2016,2017,2018]
		self.df = None

	def write(self,string):
		sys.stdout.write(string)
		sys.stdout.flush()

	def load_airport(self):
		header = ["Date","Airline Code","Airline Name","Call Sign","Movement LSV","AC Type","IATA","ICAO","T actual","T expected","RWY","ATM Def","Seats"]
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
		days 			= self.df["Date"]
		times 			= self.df["T actual"]
		times_planned 	= self.df["T expected"]
		self.df['Timestamp'] 	= pd.to_datetime(days + " " + times,format=date_format,errors="coerce")
		self.df["Year"] 		= pd.to_datetime(days + " " + times,format=date_format,errors="coerce").dt.year
		self.df["Weekday"] 	= pd.to_datetime(days + " " + times,format=date_format,errors="coerce").dt.dayofweek
		if self.db:
			self.write("[OK] loading timestamps completed\n")

	def get_delay(self):
		if self.db:
			self.write("[  ] loading delay...\r")
		date_format = "%H:%M:%S"
		times 			= self.df["T actual"]
		times_planned 	= self.df["T expected"]
		self.df['Delay'] = (pd.to_datetime(times,format=date_format,errors="coerce") - pd.to_datetime(times_planned,format=date_format,errors="coerce"))
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
		header = ["Name","City","Country","IATA","ICAO","Destination Lat","Destination Long","Altitude","Timezone","DST"]
		codes_df = pd.read_csv(filename, encoding = "utf-8", skiprows=0,delimiter =",",names=header)
		codes_df.set_index("IATA",inplace=True)
		codes_df = codes_df.loc[:,"Destination Lat":"Destination Long"]
		#print("ZRH: ",codes_df.loc["ZRH",:])
		#self.df.set_index("IATA",inplace=True)
		self.df = pd.merge(self.df, codes_df, on="IATA")

	def load_weather(self,filename = "weather.csv"):
		header = ["Wetterstation","Time","Boeenspitze 3s","Boeenspitze 1s","Temp","Niederschlag","Visibility","Windgeschwindigkeit","Windrichtung"]
		if self.db:
			self.write("[  ] loading dataset...\r")
		self.weather = pd.read_csv(filename, skiprows = 1, encoding = "utf-8",delimiter =",",names=header)
		date_format 	= "%Y%m%d%H%M"
		days 			= self.weather["Time"]
		self.weather['Time'] = pd.to_datetime(days, format=date_format, errors="coerce")
		self.weather.replace('-', np.nan)
		# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		# 	print(self.weather.head())
		if self.db:
			self.write("[OK]\n")

	def interpolate_weather(self):
		if self.db:
			self.write("[  ] interpolating dataset...\r")
		w = self.weather
		#print(w["Time"].astype(np.int64)/1000000)
		#print("")
		w['Boeenspitze 1s'] = pd.to_numeric(w['Boeenspitze 1s'], errors="coerce")
		w['Boeenspitze 3s'] = pd.to_numeric(w['Boeenspitze 3s'], errors="coerce")
		w['Temp'] = pd.to_numeric(w['Temp'], errors="coerce")
		w['Niederschlag'] = pd.to_numeric(w['Niederschlag'], errors="coerce")
		w['Visibility'] = pd.to_numeric(w['Visibility'], errors="coerce")
		w['Windgeschwindigkeit'] = pd.to_numeric(w['Windgeschwindigkeit'], errors="coerce")
		w['Windrichtung'] = pd.to_numeric(w['Windrichtung'], errors="coerce")

		t = w["Time"].astype(np.int64)/1000000
		gusts1 	= interp.interp1d(t,w["Boeenspitze 1s"],fill_value="extrapolate")
		gusts3 	= interp.interp1d(t,w["Boeenspitze 3s"],fill_value="extrapolate")
		temp 	= interp.interp1d(t,w["Temp"],fill_value="extrapolate")
		precip 	= interp.interp1d(t,w["Niederschlag"],fill_value="extrapolate")
		vis 	= interp.interp1d(t,w["Visibility"],fill_value="extrapolate")
		w_sp 	= interp.interp1d(t,w["Windgeschwindigkeit"],fill_value="extrapolate")
		w_dir 	= interp.interp1d(t,w["Windrichtung"],fill_value="extrapolate")

		t = self.df['Timestamp'].astype(np.int64)/1000000
		self.df['Gusts 1s'] 		= gusts1(t)
		self.df['Gusts 3s'] 		= gusts3(t)
		self.df['Temperature'] 		= temp(t)
		self.df['Precipitation'] 	= precip(t)
		self.df['Visibility'] 		= vis(t)
		self.df['Wind Speed'] 		= w_sp(t)
		self.df['Wind Direction'] 	= w_dir(t)
		if self.db:
			self.write("[OK]\n")

	def reorder(self):
		order = ["Timestamp","Date","Year","Weekday",\
		"T expected","T actual","Delay","Airline Code",\
		"Airline Name","Call Sign","Movement LSV","RWY",\
		"ATM Def","AC Type","Wingspans","Seats","IATA","ICAO",\
		"Destination Lat","Destination Long","Gusts 1s","Gusts 3s",\
		"Temperature","Precipitation","Visibility","Wind Speed","Wind Direction"]
		self.df = self.df[order]

	def save(self):
		if self.db:
			self.write("[  ] saving dataset...\r")
		self.df.to_csv("../dataset.txt", sep=',', encoding='utf-8',index=False)
		if self.db:
			self.write("[OK]\n")
if __name__ == "__main__":

	Airport = Engine(True)
	Airport.load_airport()
	Airport.load_weather()
	Airport.get_timestamps()
	Airport.get_delay()
	Airport.get_wingspans()
	Airport.get_coordinates()
	Airport.interpolate_weather()
	Airport.reorder()
	Airport.save()

	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(Airport.df.head())








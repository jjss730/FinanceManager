"""Calculates the Efficient Fronteir for given portfolio."""

import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Portfolio:
	"""Calculates the Efficient Fronteir."""

	def __init__(self, stocks, quant, start_date, end_date, variations):
		self.stocks = stocks                   # list of stokcs in the portfolio
		self.quant = quant  # quant for each of stocks in portfolio
		self.start_date = start_date           # start date of the portfolio
		self.end_date = end_date               # end date of the portfolio
		self.variations = variations  # number of ramndom iterations
		
		self.local_returns = pd.DataFrame()
		self.local_volume = pd.DataFrame()
		self.tkr = pd.DataFrame()
		#start_date = dlg.dt_init.text()
		#end_date = dlg.dt_end.text()
		
		for i in self.stocks:
			self.tkr = yf.download(i, start = self.start_date, end = self.end_date)
			self.local_returns[i] = self.tkr['Adj Close']
			self.local_volume[i] = self.tkr['Volume']
	
		print(self.local_returns.head())
		print(self.local_returns.tail())
		#print(self.local_volume.head())
		#df_returns = local_returns.copy()
		#df_volume = local_volume.copy()
		local_log_returns = np.log(self.local_returns / self.local_returns.shift(1))
		np.random.seed(1)
		all_weights = np.zeros((self.variations, len(self.local_returns.columns)))
		ret_arr = np.zeros(self.variations)
		vol_arr = np.zeros(self.variations)
		sharpe_arr = np.zeros(self.variations)
	
		for j in range(self.variations):
			weights = np.array(np.random.random(len(
				self.local_returns.columns)))       # weights
			weights = weights / np.sum(weights)
			all_weights[j, :] = weights         # Save weights
			ret_arr[j] = np.sum((
				local_log_returns.mean() * weights * 252))   # Expected return
			vol_arr[j] = np.sqrt(np.dot(weights.T, np.dot(
				local_log_returns.cov() * 252, weights)))  # Expected volatility.
			sharpe_arr[j] = ret_arr[j] / vol_arr[j]
		
		self.max_sr_ret = ret_arr[sharpe_arr.argmax()]
		self.max_sr_vol = vol_arr[sharpe_arr.argmax()]
		self.max_sr_weights = all_weights[sharpe_arr.argmax()]
		self.max_sr = sharpe_arr[sharpe_arr.argmax()]
		self.port_weights = df_portfolio['Quant'].tolist() / df_portfolio['Quant'].sum()
		self.port_ret = np.sum((local_log_returns.mean()* self.port_weights * 252))   # Expected return
		self.port_vol = np.sqrt(np.dot(self.port_weights.T, np.dot(
			local_log_returns.cov() * 252, self.port_weights)))  # Expected volatility
		self.port_sharpe = self.port_ret / self.port_vol
		
	def set_stocks(self, stocks):
		"""Return list of stocks in the portfolio."""
		self.stocks = stocks
		
	def set_quant(self, quant):
		self.quant = quant
		
	def set_dates(self, start_date, end_date):
		self.start_date = start_date
		self.end_date = end_date
		
	def set_variations(self, var):
		self.variations = var
	
	def portfolio_weights(self):
		return self.port_weights
	
	def portfolio_returns(self):
		return self.port_ret
	
	def portfolio_volume(self):
		return self.local_volume
	
	def portfolio_volatility(self):
		return self.port_vol
	
	def portfolio_sharpe(self):
		return self.port_sharpe
	
	def max_sharpe_return(self):
		return self.max_sr_ret
	
	def max_sharpe_volatility(self):
		return self.max_sr_vol
	
	def max_sharpe_weight(self):
		return self.max_sr_weights
	
	def max_sharpe_ratio(self):
		return self.max_sr
	
		
class Canvas(FigureCanvas):
	"""Include chart in PyQT5."""
	
	def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
		fig = Figure(figsize=(width, height), dpi = dpi)
		self.axes = fig.add_subplot(111)
		
		FigureCanvas.__init__(self, fig)
		self.setParent(parent)
		
		self.plot
		
	def plot(self):
		x = np.array([50, 30, 40])
		labels = ["Apples", "Bananas", "Melons"]
		ax = self.figure.add_subplot(111)
		ax.pie(x, labels)
		
		
class AppDemo(QWidget):
	
	def __init__(self):
		super().__init__()
		self.resize(1600,800)
		
		chart = Canvas(self)


def add_item():
	ticker = dlg.txt_ticker.text()
	try:
		quant = int(dlg.txt_quant.text())
	except ValueError:
		dlg.txt_quant.setText('0')
		quant = 0

	if not(dlg.txt_ticker.text() == ''):
		last_row = dlg.tbl_portfolio.rowCount()
		dlg.tbl_portfolio.insertRow(last_row)
		ticker = ticker.upper()
		ticker = ticker.replace(" ", "")
		dlg.tbl_portfolio.setItem(last_row, 0, QTableWidgetItem(ticker))
		dlg.tbl_portfolio.setItem(last_row, 1, QTableWidgetItem(str(quant)))
		dlg.txt_ticker.setText('')
		dlg.txt_quant.setText('')
		dlg.txt_ticker.setFocus()
		
	dlg.btn_add.setEnabled(False)


def read_table():
	global df_portfolio
	rowCount = dlg.tbl_portfolio.rowCount()
	columnCount = dlg.tbl_portfolio.columnCount()
	
	temp_df = pd.DataFrame(columns=['Stocks', 'Quant'], index = range(rowCount))

	print(rowCount)
	for row in range(rowCount):
		for column in range(columnCount):
			temp_df.iloc[row, column] = dlg.tbl_portfolio.item(row, column).text()
	#print(temp_df.head())
	#print(temp_df.tail())
	temp_df['Quant'] = temp_df['Quant'].astype(int)
	df_portfolio = temp_df.copy()
	return temp_df

def clear_table():
	dlg.tbl_portfolio.clear()
	dlg.tbl_portfolio.setRowCount(0)

def ticker_box_changed():
	if dlg.txt_ticker.text() == '':
		dlg.btn_add.setEnabled(False)
	else:
		dlg.btn_add.setEnabled(True)


def fetchData():
	#global df_returns
	#global df_volume
	local_returns_df = read_table()
	# read_table()
	start_date = dlg.dt_init.text()
	end_date = dlg.dt_end.text()
	# port1 = Portfolio(df_portfolio['Stocks'], df_portfolio['Quant'], start_date, end_date, 100)
	port1 = Portfolio(local_returns_df['Stocks'], local_returns_df['Quant'], start_date, end_date, num_port)

	print('Portfolio Return: {:.2}'.format(port1.portfolio_returns()))
	print('Portfolio Volatility: {:.2}'.format(port1.portfolio_volatility()))
	print('Portfolio Sharpe Ratio: {:.2}'.format(port1.portfolio_sharpe()))
	print('Portfolio Weights: ', port1.portfolio_weights())
	print('Max Sharpe Return: {:.2}'.format(port1.max_sharpe_return()))
	print('Max Sharpe Volatility: {:.2}'.format(port1.max_sharpe_volatility()))
	print('Max Sharpe Ratio: {:.2}'.format(port1.max_sharpe_ratio()))
	print('Max Sharpe Weights: ', port1.max_sharpe_weight())
	

if __name__ == '__main__':

	df_portfolio = pd.DataFrame()
	df_returns = pd.DataFrame()
	df_volume = pd.DataFrame()
	
	app = QtWidgets.QApplication(sys.argv)
	dlg = uic.loadUi('test.ui')
	
	num_port = int(dlg.txt_iter.text())
	dlg.txt_ticker.setFocus()
	dlg.btn_add.clicked.connect(add_item)
	dlg.txt_quant.returnPressed.connect(add_item)
	dlg.txt_ticker.textChanged.connect(ticker_box_changed)

	dlg.btn_generate.clicked.connect(fetchData)
	dlg.btn_removeAll.clicked.connect(clear_table)

	dlg.show()
	app.exec()

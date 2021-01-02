"""Calculates the Efficient Fronteir for given portfolio."""

import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QMainWindow

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg\
	as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
	"""Create Main Window."""

	def __init__(self):
		super(MainWindow, self).__init__()
		uic.loadUi('test.ui', self)
		self.show()
		
#	def MyUI(self):
#		canvas = Canvas(self, width=8, height=4)
#		canvas.move(50, 0)


class NewWindow(QMainWindow):
	"""Create Result Window."""
	
	def __init__(self):
		super(NewWindow, self).__init__()
		uic.loadUi('chartWindow.ui', self)
#		self.show()
		
	def MyUI(self):
		"""Prepare Canvas for result chart."""
		canvas = Canvas(self, width=6, height=3)
		canvas.move(5, 5)


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
		self.ret_arr = np.zeros(self.variations)
		self.vol_arr = np.zeros(self.variations)
		self.sharpe_arr = np.zeros(self.variations)
	
		for j in range(self.variations):
			weights = np.array(np.random.random(len(
				self.local_returns.columns)))       # weights
			weights = weights / np.sum(weights)
			all_weights[j, :] = weights         # Save weights
			self.ret_arr[j] = np.sum((
				local_log_returns.mean() * weights * 252))   # Expected return
			self.vol_arr[j] = np.sqrt(np.dot(weights.T, np.dot(
				local_log_returns.cov() * 252, weights)))  # Expected volatility.
			self.sharpe_arr[j] = self.ret_arr[j] / self.vol_arr[j]
		
		self.max_sr_ret = self.ret_arr[self.sharpe_arr.argmax()]
		self.max_sr_vol = self.vol_arr[self.sharpe_arr.argmax()]
		self.max_sr_weights = all_weights[self.sharpe_arr.argmax()]
		self.max_sr = \
			self.sharpe_arr[self.sharpe_arr.argmax()]
		self.port_weights = \
			df_portfolio['Quant'].tolist() / df_portfolio['Quant'].sum()
		self.port_ret = np.sum((local_log_returns.
			mean() * self.port_weights * 252))   # Expected return
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
	
	def return_array(self):
		"""Provide Returns Array."""
		return self.ret_arr

	def volatility_array(self):
		"""Provide Volatility Array."""
		return self.vol_arr
	
	def sharpe_array(self):
		"""Provide Sharpe Ratio Array."""
		return self.sharpe_arr
	
		
class Canvas(FigureCanvas):
	"""Include chart in PyQT5."""
	
	def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
		fig = Figure(figsize=(width, height), edgecolor='black', dpi = dpi)
#		self.axes = fig.add_subplot(2, 2, 1)
		FigureCanvas.__init__(self, fig)

		self.setParent(parent)
		self.plot()
		
	def plot(self):
		"""Plot pie chart of portfolio distribution."""
		global df_portfolio
		global df_portfolio_sharpe
		
		# Prepare portfolio distribution pie chart
		port_y = df_portfolio['Stocks']
		port_x = df_portfolio['Quant']
		ax = self.figure.add_subplot(2, 2, 1, title = "Curerent Portfolio")
		ax.pie(port_x, labels = port_y)
		
		# Prepare highest Sharpe ratio distribution pie chart
		shrp_y = df_portfolio['Stocks']
		shrp_x = df_portfolio_sharpe
#		shrp_x = df_portfolio['Quant']
		ax2 = self.figure.add_subplot(2, 2, 2, title = "Optimized Portfolio")
		ax2.pie(shrp_x, labels = shrp_y)
		
		# Prepare Efficient Fronteir chart
		ax3 = self.figure.add_subplot(2, 2, 3,
				title = "Efficient Fronteir",
				xlabel = "Expected Volatillity",
				ylabel = "Expected Returns")
#		ax3 = self.figure.add_subplot(gs1[:1, :],
#				title = "Efficient Fronteir",
#				xlabel = "Expected Volatillity",
#				ylabel = "Expected Returns")
		ax3.scatter(df_volatility, df_returns, c= df_sharpe)

		
#class AppDemo(QWidget):
	
#	def __init__(self):
#		super().__init__()
#		self.resize(200, 200)
#		chart = Canvas(self)


def add_item():
	"""Add stock to porfolio list."""
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
	"""Read Stock table and transfers table data to df_portfolio."""
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
	"""Clear porfolio stocks table."""
	dlg.tbl_portfolio.clear()
	dlg.tbl_portfolio.setRowCount(0)


def ticker_box_changed():
	"""Dectect changes in ticker box to enable 'add' button."""
	if dlg.txt_ticker.text() == '':
		dlg.btn_add.setEnabled(False)
	else:
		dlg.btn_add.setEnabled(True)


def fetchData():
	"""Gather data from yfinance API and calls portfolio calculations."""
	global df_returns
	global df_volatility
	global df_sharpe
	global df_portfolio
	global df_portfolio_sharpe
	
	local_portfolio_df = read_table()
	# read_table()
	start_date = dlg.dt_init.text()
	end_date = dlg.dt_end.text()
	port1 = Portfolio(local_portfolio_df['Stocks'],
			local_portfolio_df['Quant'], start_date, end_date, num_port)

	print('Portfolio Return: {:.2}'.format(port1.portfolio_returns()))
	print('Portfolio Volatility: {:.2}'.format(port1.portfolio_volatility()))
	print('Portfolio Sharpe Ratio: {:.2}'.format(port1.portfolio_sharpe()))
	print('Portfolio Weights: ', port1.portfolio_weights())
	print('Max Sharpe Return: {:.2}'.format(port1.max_sharpe_return()))
	print('Max Sharpe Volatility: {:.2}'.format(port1.max_sharpe_volatility()))
	print('Max Sharpe Ratio: {:.2}'.format(port1.max_sharpe_ratio()))
	print('Max Sharpe Weights: ', port1.max_sharpe_weight())
	
	df_portfolio = local_portfolio_df.copy()
	df_returns = port1.return_array()
	df_volatility = port1.volatility_array()
	df_sharpe = port1.sharpe_array()
	df_portfolio_sharpe = port1.max_sharpe_weight()


def show_plt():
	"""Call function to plot at aditional window."""
	charWin.MyUI()
	charWin.show()
	

if __name__ == '__main__':

	df_portfolio = pd.DataFrame()
	df_portfolio_sharpe = pd.DataFrame()
	df_returns = pd.DataFrame()
	df_volatility = pd.DataFrame()
	df_sharpe = pd.DataFrame()
	
	app = QtWidgets.QApplication(sys.argv)
	dlg = MainWindow()
	charWin = NewWindow()
	
	num_port = int(dlg.txt_iter.text())
	dlg.txt_ticker.setFocus()
	dlg.btn_add.clicked.connect(add_item)
	dlg.txt_quant.returnPressed.connect(add_item)
	dlg.txt_ticker.textChanged.connect(ticker_box_changed)

	dlg.btn_generate.clicked.connect(fetchData)
	df_returns = fetchData()
	dlg.btn_removeAll.clicked.connect(clear_table)
	dlg.btn_plt.clicked.connect(show_plt)
	charWin.btn_back.clicked.connect(lambda: charWin.close())

	dlg.show()
	app.exec()

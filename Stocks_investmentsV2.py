"""Calculates the Efficient Fronteir for given portfolio."""

import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QMainWindow

import numpy as np
import pandas as pd
import yfinance as yf
#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg\
	as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import style
style.use('ggplot')


class MainWindow(QMainWindow):
	"""Create Main Window."""

	def __init__(self):
		super(MainWindow, self).__init__()
		uic.loadUi('test.ui', self)
		self.show()


class EF_Window(QWidget):
	"""Create Result Window."""
	
	def __init__(self):
		super(EF_Window, self).__init__()
		uic.loadUi('chartWindow.ui', self)
#		self.show()
		
	def Efficient_frontier_win(self):
		"""Prepare Canvas for result chart."""
		canvas = Canvas(self, width=6, height=6)
		canvas.move(10, 10)
		

class Ret_Window(QWidget):
	"""Create Result Window."""
	
	def __init__(self):
		super(Ret_Window, self).__init__()
		uic.loadUi('StockChartWindow.ui', self)
#		self.show()
		
	def Stock_chart_win(self):
		"""Prepare Canvas to show Stock charts comparizon."""
		stock_canvas = Stock_Canvas(self, width=12, height=8)
		stock_canvas.move(5, 5)


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
		self.local_log_returns = \
			np.log(self.local_returns / self.local_returns.shift(1))
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
				self.local_log_returns.mean() * weights * 252))   # Expected return
			self.vol_arr[j] = np.sqrt(np.dot(weights.T, np.dot(
				self.local_log_returns.cov() * 252, weights)))  # Expected volatility.
			self.sharpe_arr[j] = self.ret_arr[j] / self.vol_arr[j]
		
		self.max_sr_ret = self.ret_arr[self.sharpe_arr.argmax()]
		self.max_sr_vol = self.vol_arr[self.sharpe_arr.argmax()]
		self.max_sr_weights = all_weights[self.sharpe_arr.argmax()]
		self.max_sr = \
			self.sharpe_arr[self.sharpe_arr.argmax()]
		self.port_weights = \
			df_portfolio['Quant'].tolist() / df_portfolio['Quant'].sum()
		self.port_ret = np.sum((self.local_log_returns.
			mean() * self.port_weights * 252))   # Expected return
		self.port_vol = np.sqrt(np.dot(self.port_weights.T, np.dot(
			self.local_log_returns.cov() * 252, self.port_weights)))  # Expected volatility
		self.port_sharpe = self.port_ret / self.port_vol
		self.opt_returns = self.local_log_returns.dot(self.max_sr_weights)
		
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
	
	def log_returns(self):
		"""Provide dataframe containing the log return of the portfolio."""
		return self.local_log_returns
	
	def opt_returns(self):
		"""Provide dataframe containing the optimized return of the portfolio."""
		return self.opt_returns
	
		
class Canvas(FigureCanvas):
	"""Include chart in PyQT5."""
	
	def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
		fig = Figure(figsize=(width, height), dpi = dpi)
		self.gs1 = gridspec.GridSpec(
			nrows = 2,   # Set grid system for subplots
			ncols = 2,
			figure = fig,
			wspace = 0.1,  # Width spacing.expressed as a fraction of average axis width
			hspace=0.5)  # Height spacing.expressed as a fraction of average axis height
		FigureCanvas.__init__(self, fig)

		self.setParent(parent)
		self.plot()
		
	def plot(self):
		"""Plot pie chart of portfolio distribution."""
		global df_portfolio
		global df_portfolio_sharpe
		
		# Setup colors for pie chart
		colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
		currentColor = 0
		legendVars = []
# 		for i in range(len(df_portfolio.columns)):
# 			y[i] = log_returns[log_returns.columns[i]]
# 			ax.plot_date(x, y[i],
# 				color = colors[currentColor],
# 				linewidth = 0.3,
# 				linestyle = '-',
# 				marker = ',',
# 				xdate = True)
# 			legendVars.append(log_returns.columns[i])
# 			currentColor += 1
# 			if (currentColor >= len(colors)):
# 				currentColor = 0
		
		# Prepare portfolio distribution pie chart
		port_y = df_portfolio['Stocks']
		port_x = df_portfolio['Quant']
		ax = self.figure.add_subplot(self.gs1[0, 0],
			title = "Current Portfolio")  # Position graph at grid\
		#position [line 0, column 0]
		ax.pie(port_x, labels = port_y, autopct = '%1.1f%%', labeldistance = 1.2)
		
		# Prepare highest Sharpe ratio distribution pie chart
		shrp_y = df_portfolio['Stocks']
		shrp_x = df_portfolio_sharpe
		ax2 = self.figure.add_subplot(
			self.gs1[0, 1],
			title = "Optimized Portfolio")  # Position graph at grid \
		#position [line 0, column 1]
		ax2.pie(shrp_x, labels = shrp_y, autopct = '%1.1f%%', labeldistance = 1.2)
		
		# Prepare Efficient Fronteir chart
		ax3 = self.figure.add_subplot(
			self.gs1[1, :],  # Position graph at grid position [line 1, all columns]
			title = "Efficient Frontier",
			xlabel = "Expected Volatillity",
			ylabel = "Expected Returns")
		ax3.scatter(df_volatility, df_returns, s=10, c= df_sharpe)
		ax3.scatter(max_sharpe_volatility, max_sharpe_return,
			s = 200, c = 'red', marker = "*")
		ax3.scatter(portfolio_volatility, portfolio_returns,
			s = 100, c = 'red', alpha = 0.8, marker = "o")


class Stock_Canvas(FigureCanvas):
	"""Include stock comparizon chart in PyQT5."""
	
	def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
		fig = Figure(figsize=(width, height), dpi = dpi)
		self.gs1 = gridspec.GridSpec(
			nrows = 1,   # Set grid system for subplots
			ncols = 1,
			figure = fig,
			wspace = 0.1,  # Width spacing.expressed as a fraction of average axis width
			hspace=0.5)  # Height spacing.expressed as a fraction of average axis height
		FigureCanvas.__init__(self, fig)

		self.setParent(parent)
		self.plot()
		
	def plot(self):
		"""Plot pie chart of portfolio distribution."""
		global df_portfolio
		global log_returns
		
		# Plot Return charts.
		colors = ['b', 'g', 'r', 'c', 'm', 'k', 'w']
		currentColor = 0
		legendVars = []
		x = log_returns.index
		y = pd.DataFrame()
#		y = log_returns['QQQ']
#		y2 = log_returns['X']
#		y3 = log_returns['F']
		y4 = opt_return
		
		ax = self.figure.add_subplot(self.gs1[0, 0],
						title = "Returns Chart",
						ylabel = "Returns [%]",
						xlabel = "Date")  # Position graph at grid position [line 0, column 0]

		for i in range(len(log_returns.columns)):
			y[i] = log_returns[log_returns.columns[i]]
			ax.plot_date(x, y[i],
				color = colors[currentColor],
				linewidth = 0.3,
				linestyle = '-',
				marker = ',',
				xdate = True)
			legendVars.append(log_returns.columns[i])
			currentColor += 1
			if (currentColor >= len(colors)):
				currentColor = 0


#		ax.plot_date(x, y, 'g', linewidth = 0.15, xdate = True)
#		ax.plot_date(x, y2, 'b', linewidth = 0.15, xdate = True)
#		ax.plot_date(x, y3, 'r', linewidth = 0.15, xdate = True)
		
		ax.plot_date(x, y4, 'black', linewidth = 1, xdate = True)
		ax.legend(legendVars)


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
	dlg.tbl_portfolio.setHorizontalHeaderLabels(['Stocks', 'Quant'])


def ticker_box_changed():
	"""Dectect changes in ticker box to enable 'add' button."""
	if dlg.txt_ticker.text() == '':
		dlg.btn_add.setEnabled(False)
	else:
		dlg.btn_add.setEnabled(True)


def fetch_stock_return_data():
	"""Gather data from yfinance API and calls portfolio calculations."""
	global df_returns
	global df_volatility
	global df_sharpe
	global df_portfolio
	global df_portfolio_sharpe
	global max_sharpe_return
	global max_sharpe_volatility
	global portfolio_returns
	global portfolio_volatility
	global num_port
	global port_sharpe
	global log_returns
	global opt_return
	global max_sr_weights
	
	local_portfolio_df = read_table()
	# read_table()
	start_date = dlg.dt_init.text()
	end_date = dlg.dt_end.text()
	num_port = int(dlg.txt_iter.text())
	port1 = Portfolio(local_portfolio_df['Stocks'],
			local_portfolio_df['Quant'], start_date, end_date, num_port)
	log_returns = port1.log_returns()
	max_sr_weights = port1.max_sharpe_weight()
	opt_return = log_returns.dot(max_sr_weights)
	print(opt_return)
	show_returns_plt()


def fetch_EF_data():
	"""Gather data from yfinance API and calls portfolio calculations."""
	global df_returns
	global df_volatility
	global df_sharpe
	global df_portfolio
	global df_portfolio_sharpe
	global max_sharpe_return
	global max_sharpe_volatility
	global portfolio_returns
	global portfolio_volatility
	global num_port
	global port_sharpe
	global max_sharpe
	
	local_portfolio_df = read_table()
	# read_table()
	start_date = dlg.dt_init.text()
	end_date = dlg.dt_end.text()
	num_port = int(dlg.txt_iter.text())
	
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
	max_sharpe_return = port1.max_sharpe_return()
	max_sharpe_volatility = port1.max_sharpe_volatility()
	portfolio_returns = port1.portfolio_returns()
	portfolio_volatility = port1.portfolio_volatility()
	port_sharpe = port1.portfolio_sharpe()
	max_sharpe = port1.max_sharpe_ratio()
	print('Data Fetched')
	show_EF_plt()


def show_EF_plt():
	"""Call function to plot at aditional window."""
	EF_Win.Efficient_frontier_win()
	EF_Win.lbl_NumSim2.setText(str(num_port))
	EF_Win.lbl_ExpRetCur.setText("{:.1f}%".format(portfolio_returns * 100))
	EF_Win.lbl_ExpVolCur.setText("{:.1f}%".format(portfolio_volatility * 100))
	EF_Win.lbl_ExpRetPort.setText("{:.1f}%".format(max_sharpe_return * 100))
	EF_Win.lbl_ExpVolPort.setText("{:.1f}%".format(max_sharpe_volatility * 100))
	EF_Win.lbl_ExpSrCur.setText("{:.1f}%".format(port_sharpe * 100))
	EF_Win.lbl_ExpSrPort.setText("{:.1f}%".format(max_sharpe * 100))
	EF_Win.show()


def show_returns_plt():
	"""Call function to plot at aditional window."""
	Ret_Win.Stock_chart_win()
	Ret_Win.show()
	

if __name__ == '__main__':

	df_portfolio = pd.DataFrame()
	df_portfolio_sharpe = pd.DataFrame()
	df_returns = pd.DataFrame()
	df_volatility = pd.DataFrame()
	df_sharpe = pd.DataFrame()
	
	app = QtWidgets.QApplication(sys.argv)
	dlg = MainWindow()
	EF_Win = EF_Window()
	Ret_Win = Ret_Window()
	
	dlg.txt_ticker.setFocus()
	dlg.btn_add.clicked.connect(add_item)
	dlg.txt_quant.returnPressed.connect(add_item)
	dlg.txt_ticker.textChanged.connect(ticker_box_changed)

	dlg.btn_generate.clicked.connect(fetch_EF_data)
	dlg.btn_removeAll.clicked.connect(clear_table)
#	dlg.btn_plt.clicked.connect(show_EF_plt)
	EF_Win.btn_back.clicked.connect(lambda: EF_Win.close())
	Ret_Win.btn_back2.clicked.connect(lambda: Ret_Win.close())
	dlg.btn_stockReturns.clicked.connect(fetch_stock_return_data)

	dlg.show()
	app.exec()

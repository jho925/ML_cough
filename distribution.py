from pandas import *
import numpy as np
import matplotlib.pyplot as plt

def main():

	user = list(data['user_id'])
	freq = {x:user.count(x) for x in user}
	counts = list(freq.values())


	d = {x:counts.count(x) for x in counts}

	x, y = list(d.keys()),list(d.values())

	  
	fig = plt.figure(figsize = (10, 5))
	 
	plt.bar(x, y)
	 
	plt.xlabel("Number of Coughs")
	plt.ylabel("Frequency")
	plt.title("Cough Distribution Sewanee")
	plt.show()

if __name__ == '__main__':
	main()
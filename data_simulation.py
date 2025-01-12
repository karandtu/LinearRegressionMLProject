import numpy as np
import matplotlib.pyplot as plt


# original data sampled in a 2-dimensional matrix having sampled data of 100 rows
# and 1 column from standard normal deviation(where mean is 0 and standard deviation is 1)
# We code original data using method random.rand, we keep some random noise using
# method random.randn method

#generate random data
def simulate_data():
    np.random.seed(42)
    x=np.random.rand(100,1)
    y1= 4 + 3 * x + np.random.randn(100, 1)

# plotting the data
    plt.scatter(x, y1)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.title('Generated Data for Linear Regression')
    plt.show()

    return x,y1

if __name__=='__main__':
    x,y1=simulate_data()




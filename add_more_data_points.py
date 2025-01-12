import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def data_with_more_data_points(n_datapoints=100):
    np.random.seed(42)
    x=2*np.random.rand(n_datapoints,1)
    y=4+3*x+np.random.randn(n_datapoints,1)

    return x,y

def evaluate_model_with_more_data_points(n_datapoints=100):
    x,y=data_with_more_data_points(n_datapoints)
    model = LinearRegression()
    model.fit(x,y)
    slope=model.coef_[0][0]
    intercept=model.intercept_[0]
    print(f"New Data Points:{n_datapoints},Slope: {slope}, Intercept: {intercept}")

if __name__=="__main__":
     for new_datapoints in [10,40,60,100,50]:
        evaluate_model_with_more_data_points()


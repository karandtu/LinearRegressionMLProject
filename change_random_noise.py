import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def simulate_data_with_noise(noise_level=1.0):
    np.random.seed(42)
    x = 2*np.random.rand(100,1)
    y=4+3*x+noise_level*np.random.randn(100,1)

    return x,y

def evaluate_model_with_noise(noise_level):
    x,y=simulate_data_with_noise(noise_level)
    model = LinearRegression()
    model.fit(x,y)
    y_pred = model.predict(x)
    mse = mean_squared_error(y,y_pred)
    print(f"MSE: {mse}, Noise Level: {noise_level}")


if __name__ == "__main__":
    for noise_instance in [0.1,1.0,4.0,5.0]:
        evaluate_model_with_noise(noise_instance)



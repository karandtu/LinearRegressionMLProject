import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#simulate data
def simulate_data():
       x=2*np.random.rand(100,1)
       y=4+3*x+np.random.randn(100,1)
       return x,y


#explore r2_score
def explore_r2_score():
    x,y=simulate_data()
    model=LinearRegression()
    model.fit(x,y)
    r2=model.score(x,y)
    print(f" R2Score= {r2:.4f}")

if __name__=="__main__":
    explore_r2_score()

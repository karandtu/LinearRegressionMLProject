from sklearn.linear_model import LinearRegression
from data_simulation import simulate_data

#train the model
def train_model():
    x,y1=simulate_data()
    model=LinearRegression()
    model.fit(x,y1)

#print the parameters
    print("Slope(m):" ,model.coef_[0][0])
    print("Intercept(b):",model.intercept_[0])

    return model,x,y1


if __name__=="__main__":
    train_model()



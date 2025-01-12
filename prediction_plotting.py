import matplotlib.pyplot as plt
from model_training import train_model

#Make predictions and plot the predictions

def plot_predictions():
    model, x, y1 = train_model()

    #taking new data points.
    X_new = [ [0],[2] ]
    y_new = model.predict(X_new)

    #scatter and plot as per new data points
    plt.scatter(x,y1,label="Actual Data")
    plt.plot(X_new,y_new,label="Predicted Data")
    plt.xlabel("x")
    plt.ylabel("y1")
    plt.title("Linear Regression Fit on new Data")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_predictions()



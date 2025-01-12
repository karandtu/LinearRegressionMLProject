from sklearn.metrics import mean_squared_error
from model_training import train_model

#evaluate the model now
def evaluate_model():
     model,x,y1=train_model()

     #take the same data that you used for training the model
     y_train_pred = model.predict(x)

     #compute the mean_squared_error
     mse=mean_squared_error(y1,y_train_pred)
     print("Mean Squared Error:", mse)


if __name__=="__main__":
    evaluate_model()
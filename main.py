import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


data = pd.read_csv("student_data.csv")
X=data[['Hours_Studied']]
y=data['Marks']

model = LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R²:", r2_score(y_test, y_pred))

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted marks:", y_pred)
print("Actual marks:", y_test.values)
plt.scatter(
    data["Hours_Studied"],
    data["Marks"],
    color="blue",
    label="Actual Marks"
)
X_line = X.sort_values(by="Hours_Studied")
y_line = model.predict(X_line)

plt.plot(
    X_line["Hours_Studied"],
    y_line,
    color="red",
    label="Regression Line"
)
plt.xlabel("Hours Studied")
plt.ylabel("marks out of 100 ")
plt.title("Student Marks Prediction")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

plt.show()
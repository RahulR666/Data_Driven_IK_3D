import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split
from keras import models, layers, regularizers

l1 = l2 = l3 = l4 = l5 = l6 = l7 = 2.5

def seven_link_manipulator_dataset():
    rows = []
    for i in range(0, 55000):
        theta1 = round(random.uniform(0, math.pi), 2)
        theta2 = round(random.uniform(-math.pi, 0), 2)
        theta3 = round(random.uniform(-math.pi/2, math.pi/2), 2)
        theta4 = round(random.uniform(0, math.pi), 2)
        theta5 = round(random.uniform(-math.pi, 0), 2)
        theta6 = round(random.uniform(-math.pi/2, math.pi/2), 2)
        theta7 = round(random.uniform(-math.pi, math.pi/2), 2)

        x = round(l1*math.cos(theta1)+l2*math.cos(theta1+theta2)+l3*math.cos(theta1+theta2+theta3)+l4*math.cos(theta1+theta2+theta3+theta4)+l5*math.cos(theta1+theta2+theta3+theta4+theta5)+l6*math.cos(theta1+theta2+theta3+theta4+theta5+theta6)+l7*math.cos(theta1+theta2+theta3+theta4+theta5+theta6+theta7), 2)
        y = round(l1*math.sin(theta1)+l2*math.sin(theta1+theta2)+l3*math.sin(theta1+theta2+theta3)+l4*math.sin(theta1+theta2+theta3+theta4)+l5*math.sin(theta1+theta2+theta3+theta4+theta5)+l6*math.sin(theta1+theta2+theta3+theta4+theta5+theta6)+l7*math.sin(theta1+theta2+theta3+theta4+theta5+theta6+theta7), 2)
        phi = round(math.degrees(theta1)+math.degrees(theta2)+math.degrees(theta3)+math.degrees(theta4)+math.degrees(theta5)+math.degrees(theta6)+math.degrees(theta7), 2)

        rows.append([theta1, theta2, theta3, theta4, theta5, theta6, theta7, x, y, phi])

    df = pd.DataFrame(rows, columns=['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'x', 'y', 'phi'])
    df.to_csv('seven_link.csv', index=False)

seven_link_manipulator_dataset()

seven_link_data = pd.read_csv('seven_link.csv')
seven_link_data.head(50000)

plt.scatter(seven_link_data['x'], seven_link_data['y'])
plt.show()

X = seven_link_data[['x', 'y', 'phi']]
y = seven_link_data[['theta1', 'theta2', 'theta3', 'theta4' , 'theta5', 'theta6','theta7']]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15)

model = models.Sequential()
model.add(layers.Dense(units=7, input_dim=3, kernel_initializer='uniform'))
model.add(layers.Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(layers.Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(layers.Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(layers.Dense(units=7, kernel_initializer = 'uniform', activation='linear'))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#Training model
history = model.fit(train_X, train_y, epochs=150, validation_split=0.15, shuffle=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

test_loss, test_acc = model.evaluate(test_X, test_y)

print('\nTesting Loss = ', test_loss)
print('Testing Accuracy = ', test_acc)

from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(train_X, train_y)

# Predict using the model
rf_predictions = rf_model.predict(test_X)


from sklearn.tree import DecisionTreeRegressor

# Create a Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(train_X, train_y)

# Predict using the model
dt_predictions = dt_model.predict(test_X)


from sklearn.metrics import mean_squared_error, r2_score

# Calculate predictions for all models
rf_predictions = rf_model.predict(test_X)
dt_predictions = dt_model.predict(test_X)
nn_predictions = model.predict(test_X)

# Calculate Mean Squared Errors
rf_mse = mean_squared_error(test_y, rf_predictions)
dt_mse = mean_squared_error(test_y, dt_predictions)
nn_mse = mean_squared_error(test_y, nn_predictions)

# Calculate R-squared (R2) scores
rf_r2 = r2_score(test_y, rf_predictions)
dt_r2 = r2_score(test_y, dt_predictions)
nn_r2 = r2_score(test_y, nn_predictions)

# Display results
print("Random Forest:")
print("Mean Squared Error:", rf_mse)
print("R-squared (R2) Score:", rf_r2)
print()

print("Decision Tree:")
print("Mean Squared Error:", dt_mse)
print("R-squared (R2) Score:", dt_r2)
print()


print("Neural Network:")
print("Mean Squared Error:", nn_mse)
print("R-squared (R2) Score:", nn_r2)


import matplotlib.pyplot as plt

# Model names
models = ['Random Forest', 'Decision Tree', 'Neural Network']

# Mean Squared Errors
mse_scores = [rf_mse, dt_mse , nn_mse]

# R-squared (R2) Scores
r2_scores = [rf_r2, dt_r2, nn_r2]

# Plot Mean Squared Errors
plt.figure(figsize=(10, 6))
plt.bar(models, mse_scores, color='blue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Mean Squared Error for Different Models')
plt.ylim(min(mse_scores) - 0.01, max(mse_scores) + 0.01)
plt.tight_layout()
plt.show()

# Plot R-squared (R2) Scores
plt.figure(figsize=(10, 6))
plt.bar(models, r2_scores, color='green')
plt.xlabel('Models')
plt.ylabel('R-squared (R2) Score')
plt.title('Comparison of R-squared (R2) Score for Different Models')
plt.ylim(min(r2_scores) - 0.01, max(r2_scores) + 0.01)
plt.tight_layout()
plt.show()


def seven_link_forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6,theta7):
    x = (np.cos(theta1)*l1)+(np.cos(theta1+theta2)*l2)+(np.cos(theta1+theta2+theta3)*l3)+(np.cos(theta1+theta2+theta3+theta4)*l4)+(np.cos(theta1+theta2+theta3+theta4+theta5)*l5)+(np.cos(theta1+theta2+theta3+theta4+theta5+theta6)*l6)+(np.cos(theta1+theta2+theta3+theta4+theta5+theta6+theta7)*l7)
    y = (np.sin(theta1)*l1)+(np.sin(theta1+theta2)*l2)+(np.sin(theta1+theta2+theta3)*l3)+(np.sin(theta1+theta2+theta3+theta4)*l4)+(np.sin(theta1+theta2+theta3+theta4+theta5)*l5)+(np.sin(theta1+theta2+theta3+theta4+theta5+theta6)*l6)+(np.sin(theta1+theta2+theta3+theta4+theta5+theta6+theta7)*l7)

    return x, y


def iiitm_dataset():
    rows = []

    # for I
    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = 6
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    # for I
    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = 7
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    # for I
    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = 8
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    # for T
    arr = np.linspace(8.5, 10.5, 100)
    for x in arr:
        y = 7
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = 9.5
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    # for M
    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = 11
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = ((-y)+29)/2
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = ((y)+19)/2
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    arr = np.linspace(5, 7, 100)
    for y in arr:
        x = 13
        theta = math.degrees(math.atan(y/x))
        rows.append([x, y, theta])

    df = pd.DataFrame(rows, columns=['x', 'y', 'theta'])
    df.to_csv('iiitm.csv', index=False)

iiitm_dataset()

data = pd.read_csv('iiitm.csv')
data.head()

iiitm_data = pd.read_csv('iiitm.csv')
iiitm_predictions = model.predict(iiitm_data.values)
print(iiitm_predictions)

iiitm_data = pd.read_csv('iiitm.csv')
iiitm_predictions = model.predict(iiitm_data.values)
x_predictions, y_predictions = seven_link_forward_kinematics(iiitm_predictions[:,0], iiitm_predictions[:,1], iiitm_predictions[:,2], iiitm_predictions[:,3], iiitm_predictions[:,4], iiitm_predictions[:,5],iiitm_predictions[:,6])

iiitm_data = pd.read_csv('iiitm.csv')
iiitm_predictions = model.predict(iiitm_data.values)
dt_predictions = dt_model.predict(iiitm_data.values)
rf_predictions = rf_model.predict(iiitm_data.values)
x_predictions, y_predictions = seven_link_forward_kinematics(iiitm_predictions[:,0], iiitm_predictions[:,1], iiitm_predictions[:,2], iiitm_predictions[:,3], iiitm_predictions[:,4], iiitm_predictions[:,5], iiitm_predictions[:,6])
x_rf_predictions, y_rf_predictions = seven_link_forward_kinematics(rf_predictions[:, 0], rf_predictions[:, 1],rf_predictions[:, 2],rf_predictions[:, 3], rf_predictions[:, 4],rf_predictions[:, 5],rf_predictions[:, 6])
x_dt_predictions, y_dt_predictions = seven_link_forward_kinematics(dt_predictions[:, 0], dt_predictions[:, 1], dt_predictions[:, 2],dt_predictions[:, 3], dt_predictions[:, 4], dt_predictions[:, 5],dt_predictions[:, 6])


# Plot Original and Predicted for each model
plt.plot(iiitm_data['x'], iiitm_data['y'], 'go', label='Original')
plt.show()
plt.plot(x_predictions, y_predictions, 'yo', label = 'Predicted(NN)')
plt.legend()
plt.show()
plt.plot(x_rf_predictions, y_rf_predictions, 'bo', label='Random Forest')
plt.legend()
plt.show()
plt.plot(x_dt_predictions, y_dt_predictions, 'ro', label='Decision Tree')
plt.legend()
plt.show()

plt.plot(iiitm_data['x'], iiitm_data['y'], 'go', label='Original')
plt.plot(x_predictions, y_predictions, 'yo', label = 'Predicted(NN)')
plt.plot(x_rf_predictions, y_rf_predictions, 'bo', label='Random Forest')
plt.plot(x_dt_predictions, y_dt_predictions, 'ro', label='Decision Tree')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

# Calculate predictions using FABRIK solver
fabrik_predictions = np.zeros_like(iiitm_data[['x', 'y']].values)
for i, row in iiitm_data.iterrows():
    target_position = np.array([row['x'], row['y']])
    nn_predicted_angles = iiitm_predictions[i]  # Using Neural Network predicted angles
    fabrik_predicted_position = fabrik_solver(link_lengths, target_position, nn_predicted_angles)
    fabrik_predictions[i] = fabrik_predicted_position

# Calculate Mean Squared Error and R-squared (R2) Score for FABRIK predictions
fabrik_mse = mean_squared_error(iiitm_data[['x', 'y']].values, fabrik_predictions)
fabrik_r2 = r2_score(iiitm_data[['x', 'y']].values, fabrik_predictions)

print("FABRIK:")
print("Mean Squared Error:", fabrik_mse)
print("R-squared (R2) Score:", fabrik_r2)


from sklearn.metrics import mean_squared_error, r2_score

# Calculate predictions using Neural Network
nn_predictions = model.predict(iiitm_data.values)
x_nn_predictions, y_nn_predictions = seven_link_forward_kinematics(nn_predictions[:,0], nn_predictions[:,1], nn_predictions[:,2], nn_predictions[:,3], nn_predictions[:,4], nn_predictions[:,5], nn_predictions[:,6])

# Calculate predictions using Decision Tree
dt_predictions = dt_model.predict(iiitm_data.values)
x_dt_predictions, y_dt_predictions = seven_link_forward_kinematics(dt_predictions[:,0], dt_predictions[:,1], dt_predictions[:,2], dt_predictions[:,3], dt_predictions[:,4], dt_predictions[:,5], dt_predictions[:,6])

# Calculate predictions using Random Forest
rf_predictions = rf_model.predict(iiitm_data.values)
x_rf_predictions, y_rf_predictions = seven_link_forward_kinematics(rf_predictions[:,0], rf_predictions[:,1], rf_predictions[:,2], rf_predictions[:,3], rf_predictions[:,4], rf_predictions[:,5], rf_predictions[:,6])

# Calculate Mean Squared Error and R-squared (R2) scores for Neural Network
nn_mse = mean_squared_error(iiitm_data[['x', 'y']].values, np.column_stack((x_nn_predictions, y_nn_predictions)))
nn_r2 = r2_score(iiitm_data[['x', 'y']].values, np.column_stack((x_nn_predictions, y_nn_predictions)))

# Calculate Mean Squared Error and R-squared (R2) scores for Decision Tree
dt_mse = mean_squared_error(iiitm_data[['x', 'y']].values, np.column_stack((x_dt_predictions, y_dt_predictions)))
dt_r2 = r2_score(iiitm_data[['x', 'y']].values, np.column_stack((x_dt_predictions, y_dt_predictions)))

# Calculate Mean Squared Error and R-squared (R2) scores for Random Forest
rf_mse = mean_squared_error(iiitm_data[['x', 'y']].values, np.column_stack((x_rf_predictions, y_rf_predictions)))
rf_r2 = r2_score(iiitm_data[['x', 'y']].values, np.column_stack((x_rf_predictions, y_rf_predictions)))

print("Neural Network:")
print("Mean Squared Error:", nn_mse)
print("R-squared (R2) Score:", nn_r2)
print()

print("Decision Tree:")
print("Mean Squared Error:", dt_mse)
print("R-squared (R2) Score:", dt_r2)
print()

print("Random Forest:")
print("Mean Squared Error:", rf_mse)
print("R-squared (R2) Score:", rf_r2)

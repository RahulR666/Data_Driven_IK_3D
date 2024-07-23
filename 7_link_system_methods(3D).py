import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split
from keras import models, layers, regularizers

l1 = l2 = l3 = l4 = l5 = l6 = l7 = 1.25
a1 = a2= a3 = a4 = a5 = a6= a7= 0.5

def seven_link_manipulator_dataset():
    rows = []
    for i in range(0, 50000):
        theta1 = round(random.uniform(0, math.pi), 2)
        theta2 = round(random.uniform(-math.pi, 0), 2)
        theta3 = round(random.uniform(-math.pi/2, math.pi/2), 2)
        theta4 = round(random.uniform(0, math.pi), 2)
        theta5 = round(random.uniform(-math.pi, 0), 2)
        theta6 = round(random.uniform(-math.pi/2, math.pi/2), 2)
        theta7 = round(random.uniform(0,math.pi),2)

        h0 = np.array([[1, 0 , 0 , 0],
              [0, math.cos(theta1), -math.sin(theta1),0],
              [0, math.sin(theta1), math.cos(theta1), 0],
              [0, 0, 0, 1]])

        h1 = np.array([[1, 0 , 0 , l1],
              [0, math.cos(theta2), -math.sin(theta2) ,l1 ],
              [0, math.sin(theta2), math.cos(theta2), a1],
              [0, 0, 0, 1]])

        h2 = np.array([[math.cos(theta3), -math.sin(theta3), 0 , l1+l2],
              [math.sin(theta3), math.cos(theta3), 0 , l1+l2],
              [0, 0, 1, a1+a2],
              [0, 0, 0, 1]])

        h3 = np.array([[math.cos(theta4), -math.sin(theta4), 0 , l1+l2+l3],
              [math.sin(theta4), math.cos(theta4), 0 ,  l1+l2+l3],
              [0, 0, 1, a1+a2+a3],
              [0, 0, 0, 1]])

        h4 = np.array([[math.cos(theta5), 0, -math.sin(theta5) , l1+l2+l3+l4],
              [0, 1 , 0 , l1+l2+l3+l4],
              [math.sin(theta5), 0, math.cos(theta5), a1+a2+a3+a4],
              [0, 0, 0, 1]])

        h5 = np.array([[math.cos(theta6), 0, -math.sin(theta6) , l1+l2+l3+l4+l5],
              [0, 1 , 0 , l1+l2+l3+l4],
              [math.sin(theta6), 0, math.cos(theta6), a1+a2+a3+a4+a5],
              [0, 0, 0, 1]])

        h6 = np.array([[math.cos(theta7), 0, -math.sin(theta7) , l1+l2+l3+l4+l6],
              [0, 1 , 0 , l1+l2+l3+l4],
              [math.sin(theta7), 0, math.cos(theta7), a1+a2+a3+a4+a5+a6],
              [0, 0, 0, 1]])

        h7 = np.array([[1, 0, 0 , l1+l2+l3+l4+l5+l6+l7],
              [0, 1 , 0, l1+l2+l3+l4+l5+l6+l7],
              [0, 0, 1, a1+a2+a3+a4+a5+a6+a7],
              [0, 0, 0, 1]])

        h01 = np.dot(h0,h1)
        h02 = np.dot(h01,h2)
        h03 = np.dot(h02,h3)
        h04 = np.dot(h03,h4)
        h05 = np.dot(h04,h5)
        h06 = np.dot(h05,h6)
        h07 = np.dot(h06,h7)
        # print(h01)
        # print(h02)
        # print(h03)
        x_ = h07[0:1,3]
        x = float(x_)
        y_ = h07[1:2,3]
        y = float(y_)
        z_ = h07[2:3,3]
        z = float(z_)
        t1 = h07[0, 0:1]
        t1 = float(t1)
        # print(type(t1))
        t2 = h07[0, 1:2]
        t2 = float(t2)
        t3 = h07[0, 2:3]
        t3 = float(t3)
        t4 = h07[1, 0:1]
        t4 = float(t4)
        t5 = h07[1, 1:2]
        t5 = float(t5)
        t6 = h07[1, 2:3]
        t6 = float(t6)
        t7 = h07[2, 0:1]
        t7 = float(t7)
        t8 = h07[2, 1:2]
        t8 = float(t8)
        t9 = h07[2, 2:3]
        t9 = float(t9)
        # phi_ = h03[0:3, 0:3]
        # phi = phi_.astype(float)
        # print(x)
        # print(y)
        # print(z)
        # print(phi)

        rows.append([theta1, theta2, theta3,theta4, theta5, theta6, theta7, x, y, z, t1, t2, t3 ,t4, t5 ,t6, t7, t8, t9])
    df = pd.DataFrame(rows, columns=['theta1', 'theta2', 'theta3' , 'theta4', 'theta5', 'theta6','theta7', 'x', 'y', 'z' ,'t1', 't2', 't3' ,'t4', 't5' ,'t6', 't7', 't8', 't9' ])
    df.to_csv('seven_link.csv', index=False)

seven_link_manipulator_dataset()

seven_link_data = pd.read_csv('seven_link.csv')
seven_link_data.head(50000)

ax = plt.axes(projection ='3d')
ax.scatter3D(seven_link_data['x'], seven_link_data['y'], seven_link_data['z'], color = "blue")
plt.title("simple 3D scatter plot")
plt.show()

X = seven_link_data[['x', 'y', 'z', 't1', 't2', 't3' ,'t4', 't5' ,'t6', 't7', 't8', 't9' ]]
Y = seven_link_data[['theta1', 'theta2', 'theta3','theta4', 'theta5', 'theta6','theta7']]
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)

model = models.Sequential()
model.add(layers.Dense(units=12, input_dim=12, kernel_initializer='uniform'))
model.add(layers.Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(layers.Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(layers.Dense(units=7, kernel_initializer = 'uniform', activation='linear'))

model.summary()


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#Training model
history = model.fit(train_X, train_Y, epochs=100, validation_split=0.1, shuffle=True)

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

test_loss, test_acc = model.evaluate(test_X, test_Y)

print('\nTesting Loss = ', test_loss)
print('Testing Accuracy = ', test_acc)

from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(train_X, train_Y)

# Predict using the model
rf_predictions = rf_model.predict(test_X)

from sklearn.tree import DecisionTreeRegressor

# Create a Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(train_X, train_Y)

# Predict using the model
dt_predictions = dt_model.predict(test_X)

from sklearn.metrics import mean_squared_error, r2_score

# Calculate predictions for all models
rf_predictions = rf_model.predict(test_X)
dt_predictions = dt_model.predict(test_X)
nn_predictions = model.predict(test_X)

# Calculate Mean Squared Errors
rf_mse = mean_squared_error(test_Y, rf_predictions)
dt_mse = mean_squared_error(test_Y, dt_predictions)
nn_mse = mean_squared_error(test_Y, nn_predictions)

# Calculate R-squared (R2) scores
rf_r2 = r2_score(test_Y, rf_predictions)
dt_r2 = r2_score(test_Y, dt_predictions)
nn_r2 = r2_score(test_Y, nn_predictions)

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


def seven_link_forward_kinematics(theta1, theta2, theta3,theta4,theta5,theta6,theta7):
  h0 = np.array([[0, np.cos(theta1), -np.sin(theta1), 0],
              [0, np.sin(theta1), np.cos(theta1) , 0 ],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

  h1 = np.array([[0, np.cos(theta2), -np.sin(theta2), l1],
              [0, np.sin(theta2), np.cos(theta2), l1],
              [0, 0, 1, a1],
              [0, 0, 0, 1]])

  h2 = np.array([[np.sin(theta3) ,np.cos(theta3), 0 , l1+l2],
              [np.cos(theta3), -np.sin(theta3), 0, l1+l2 ],
              [0, 0 , 1 , a1+a2],
              [0, 0, 0, 1]])

  h3 = np.array([[np.sin(theta4) ,np.cos(theta4), 0 , l1+l2+l3],
              [ np.cos(theta4), -np.sin(theta4),0 , l1+l2+l3],
              [0,0 , 1 , a1+a2],
              [0, 0, 0, 1]])

  h4 = np.array([[np.cos(theta5),0, -np.sin(theta5) , l1+l2+l3+l4],
              [np.sin(theta5),0, np.cos(theta5) , l1+l2+l3+l4],
              [0, 1, 0 , a1+a2+a3+a4],
              [0, 0, 0, 1]])

  h5 = np.array([[np.cos(theta6),0,  -np.sin(theta6), l1+l2+l3+l4+l5],
              [np.sin(theta6), 0, np.cos(theta6) , l1+l2+l3+l4+l5 ],
              [0, 1, 0 , a1+a2+a3+a4+a5],
              [0, 0, 0, 1]])
  h6 = np.array([[np.cos(theta7),0,  -np.sin(theta7), l1+l2+l3+l4+l5+l6],
              [np.sin(theta7),0,  np.cos(theta7) ,l1+l2+l3+l4+l5+l6],
              [0, 1, 0, a1+a2+a3+a4+a5+a6],
              [0, 0, 0, 1]])

  h7 = np.array([[1, 0, 0 , l1+l2+l3+l4+l5+l6],
              [0, 1 , 0,l1+l2+l3+l4+l5+l6],
              [0, 0, 1, a1+a2+a3+a4+a5+a6],
              [0, 0, 0, 1]])

  h01 = np.dot(h0,h1)
  h02 = np.dot(h01,h2)
  h03 = np.dot(h02,h3)
  h04 = np.dot(h03,h4)
  h05 = np.dot(h04,h5)
  h06 = np.dot(h05,h6)
  h07 = np.dot(h06,h7)
  x = h07[0:1,3]
  # x = float(x)
  y = h07[1:2,3]
  # y = float(y)
  z = h07[2:3,3]
  # z = float(z)
  t1 = h06[0, 0:1]
  # t1 = float(t1)
  # print(type(t1))
  t2 = h06[0, 1:2]
  # t2 = float(t2)
  t3 = h06[0, 2:3]
  # t3 = float(t3)
  t4 = h06[1, 0:1]
  # t4 = float(t4)
  t5 = h06[1, 1:2]
  # t5 = float(t5)
  t6 = h06[1, 2:3]
  # t6 = float(t6)
  t7 = h06[2, 0:1]
  # t7 = float(t7)
  t8 = h06[2, 1:2]
  # t8 = float(t8)
  t9 = h06[2, 2:3]
  # t9 = float(t9)

  return x, y, z


iiitm_data = pd.read_csv('iiitm_3d_orientation.csv')
iiitm_predictions = model.predict(iiitm_data.values)
dt_predictions = dt_model.predict(iiitm_data.values)
rf_predictions = rf_model.predict(iiitm_data.values)

x_predictions, y_predictions, z_predictions = seven_link_forward_kinematics(iiitm_predictions[:,0], iiitm_predictions[:,1], iiitm_predictions[:,2],iiitm_predictions[:,3], iiitm_predictions[:,4], iiitm_predictions[:,5],iiitm_predictions[:,6])
x_rf_predictions, y_rf_predictions , z_rf_predictions = seven_link_forward_kinematics(rf_predictions[:, 0], rf_predictions[:, 1],rf_predictions[:, 2],rf_predictions[:, 3], rf_predictions[:, 4],rf_predictions[:, 5],rf_predictions[:, 6])
x_dt_predictions, y_dt_predictions,z_dt_predictions = seven_link_forward_kinematics(dt_predictions[:, 0], dt_predictions[:, 1], dt_predictions[:, 2],dt_predictions[:, 3], dt_predictions[:, 4], dt_predictions[:, 5], dt_predictions[:, 6])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iiitm_data['x'], iiitm_data['y'], iiitm_data['z'], c='g', marker='o', label='Original')
for i in range(len(x_predictions)):
    ax.scatter(x_predictions[i], y_predictions[i], z_predictions[i], c='y', marker='o', label='Predicted(NN)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iiitm_data['x'], iiitm_data['y'], iiitm_data['z'], c='g', marker='o', label='Original')
for i in range(len(x_rf_predictions)):
    ax.scatter(x_rf_predictions[i], y_rf_predictions[i], z_rf_predictions[i], c='b', marker='o', label='Random Forest')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iiitm_data['x'], iiitm_data['y'], iiitm_data['z'], c='g', marker='o', label='Original')
# Plot Predicted Data Points for Decision Tree
for i in range(len(x_dt_predictions)):
    ax.scatter(x_dt_predictions[i], y_dt_predictions[i], z_dt_predictions[i], c='r', marker='o', label='Decision Tree')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Original Data Points
ax.scatter(iiitm_data['x'], iiitm_data['y'], iiitm_data['z'], c='g', marker='o', label='Original')

# Plot Predicted Data Points for Neural Network
for i in range(len(x_predictions)):
    ax.scatter(x_predictions[i], y_predictions[i], z_predictions[i], c='y', marker='o', label='Predicted(NN)')

# Plot Predicted Data Points for Random Forest
for i in range(len(x_rf_predictions)):
    ax.scatter(x_rf_predictions[i], y_rf_predictions[i], z_rf_predictions[i], c='b', marker='o', label='Random Forest')

# Plot Predicted Data Points for Decision Tree
for i in range(len(x_dt_predictions)):
    ax.scatter(x_dt_predictions[i], y_dt_predictions[i], z_dt_predictions[i], c='r', marker='o', label='Decision Tree')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()


import numpy as np

def fabrik_inverse_kinematics_7d(target_positions, link_lengths, initial_guess=None, max_iterations=50, tolerance=1e-6):
    num_links = len(link_lengths)

    if initial_guess is None:
        initial_guess = np.zeros((num_links, 3))

    joint_positions = initial_guess.copy()

    for iteration in range(max_iterations):
        # Forward reaching
        joint_positions[-1] = target_positions
        for i in range(num_links - 2, -1, -1):
            current_link = joint_positions[i + 1] - joint_positions[i]
            target_link = target_positions - joint_positions[i]
            joint_positions[i] = joint_positions[i + 1] - (link_lengths[i] / np.linalg.norm(current_link)) * target_link

        # Backward reaching
        joint_positions[0] = initial_guess[0]
        for i in range(num_links - 1):
            current_link = joint_positions[i + 1] - joint_positions[i]
            target_link = joint_positions[i + 1] - joint_positions[i]
            joint_positions[i + 1] = joint_positions[i] + (link_lengths[i] / np.linalg.norm(current_link)) * target_link

        # Check convergence
        if np.all(np.abs(joint_positions[-1] - target_positions) < tolerance):
            break

    return joint_positions



# Generate FABRIK-predicted joint positions
fabrik_predictions = []
for i in range(len(iiitm_data)):
    x, y, z = iiitm_data.iloc[i]['x'], iiitm_data.iloc[i]['y'], iiitm_data.iloc[i]['z']
    target_positions = np.array([x, y, z])
    fabrik_prediction = fabrik_inverse_kinematics_7d(target_positions, [l1, l2, l3,l4,l5,l6,l7])
    fabrik_predictions.append(fabrik_prediction)
fabrik_predictions = np.array(fabrik_predictions)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Original Data Points
ax.scatter(iiitm_data['x'], iiitm_data['y'], iiitm_data['z'], c='g', marker='o', label='Original')

# Plot Predicted Data Points for Neural Network
for i in range(len(x_predictions)):
    ax.scatter(x_predictions[i], y_predictions[i], z_predictions[i], c='y', marker='o', label='Predicted(NN)')

# Plot Predicted Data Points for Random Forest
for i in range(len(x_rf_predictions)):
    ax.scatter(x_rf_predictions[i], y_rf_predictions[i], z_rf_predictions[i], c='b', marker='o', label='Random Forest')

# Plot Predicted Data Points for Decision Tree
for i in range(len(x_dt_predictions)):
    ax.scatter(x_dt_predictions[i], y_dt_predictions[i], z_dt_predictions[i], c='r', marker='o', label='Decision Tree')

# Plot FABRIK Predicted Data Points
ax.scatter(fabrik_predictions[:, 0], fabrik_predictions[:, 1], fabrik_predictions[:, 2], c='m', marker='o', label='FABRIK')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()

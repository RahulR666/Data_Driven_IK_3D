import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from sklearn.model_selection import train_test_split
from keras import models, layers, regularizers

l1 = l2 = l3 = l4 = l5 = l6 = l7 = 2.5
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

        h0 = np.array([[math.cos(theta1), -math.sin(theta1), 0 , 0],
              [math.sin(theta1), math.cos(theta1), 0 , 0 ],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

        h1 = np.array([[math.cos(theta2), -math.sin(theta2), 0 , l1],
              [math.sin(theta2), math.cos(theta2), 0 , 0 ],
              [0, 0, 1, a1],
              [0, 0, 0, 1]])

        h2 = np.array([[math.cos(theta3), -math.sin(theta3), 0 , l1+l2],
              [math.sin(theta3), math.cos(theta3), 0 , 0 ],
              [0, 0, 1, a1+a2],
              [0, 0, 0, 1]])

        h3 = np.array([[math.cos(theta4), -math.sin(theta4), 0 , l1+l2+l3],
              [math.sin(theta4), math.cos(theta4), 0 , 0 ],
              [0, 0, 1, a1+a2+a3],
              [0, 0, 0, 1]])

        h4 = np.array([[math.cos(theta5), -math.sin(theta5), 0 , l1+l2+l3+l4],
              [math.sin(theta5), math.cos(theta5), 0 , 0 ],
              [0, 0, 1, a1+a2+a3+a4],
              [0, 0, 0, 1]])

        h5 = np.array([[math.cos(theta6), -math.sin(theta6), 0 , l1+l2+l3+l4+l5],
              [math.sin(theta6), math.cos(theta6), 0 , 0 ],
              [0, 0, 1, a1+a2+a3+a4+a5],
              [0, 0, 0, 1]])

        h6 = np.array([[math.cos(theta7), -math.sin(theta7), 0 , l1+l2+l3+l4+l5+l6],
              [math.sin(theta7), math.cos(theta7), 0 , 0 ],
              [0, 0, 1, a1+a2+a3+a4+a5+a6],
              [0, 0, 0, 1]])

        h7 = np.array([[1, 0, 0 , l1+l2+l3+l4+l5+l6+l7],
              [0, 1 , 0, 0 ],
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

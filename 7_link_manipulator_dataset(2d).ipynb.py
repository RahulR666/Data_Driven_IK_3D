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

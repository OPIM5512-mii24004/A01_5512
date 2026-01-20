#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Generate box plot
plt.boxplot(df['HouseAge'])
plt.xlabel('Data')
plt.ylabel('House Age (years)')
plt.grid(True)

plt.savefig('HouseAge_boxplot.png')

plt.show()


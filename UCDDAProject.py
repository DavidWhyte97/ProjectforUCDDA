import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Data set as Pandas Data Frame
redwine = pd.read_csv("winequality-red.csv")
#print dataframe head
print(redwine.head())
#check for data which may be missing
redwine.isnull().sum()

#sort wines based on pH in descending order (most basic first)
redwine["pH"].sort_values(ascending=False)
#index data frame by alcohol content and sort in an ascedning order
redwine_alc= redwine.set_index("alcohol")
redwine_alc.sort_index()

#obtain mean alcohol content
redwine.alcohol.mean()
# obtain alcohol content grouped by pH.
rwpH_alcohol = redwine.groupby("pH").alcohol.mean()
print(rwpH_alcohol)

#use loc to get all rows and the first column
redwine.loc[0,:]
rwsulph= redwine["sulphates"].sort_index()
rwsulph.iloc[:100]

# use itterrows
for alcohol, row in redwine.iterrows():
    print(alcohol, row[0], row[1], row[2], row[3],)

# creating new dictionary and converting it to a dataframe
newdict = {"alcohol": [9 , 7, 9.7], "pH": [3, 3.6, 4]}
print(newdict)
newwine= pd.DataFrame(newdict)
print(newwine)

#merging dataframes
newdf= redwine.merge(newwine, on=["alcohol", "pH"], how="left")
print(newdf.head())

#check for missing data on new data frame
newdf.isnull().sum()

#revert to using dataframe redwine for accuracy
#using Numpy, convert dataframe to Numpy array
import numpy as np
redwine_np = np.array(redwine)

#getting max and min alcohol levels
print(np.max(redwine["alcohol"]))
print(np.min(redwine["alcohol"]))
print(np.mean(redwine["alcohol"]))
print(np.median(redwine["alcohol"]))

#using MatPlotLib to visualise Data
# using a simple linegraph to plot any correlation between pH and alcohol
x=(list(redwine["alcohol"]))
y=(list(redwine["pH"]))

plt.plot(x, y)
plt.show()

#using a scatter plot instead with labelled axis
plt.xlabel("Alcohol")
plt.ylabel("pH")
plt.scatter(x, y)
plt.show()
























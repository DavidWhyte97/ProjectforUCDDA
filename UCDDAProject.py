import pandas as pd
import seaborn as sns
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

# # use itterrows
# for alcohol, row in redwine.iterrows():
#     print(alcohol, row[0], row[1], row[2], row[3],)

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

def print_stat_for_alcohol(stat):
    np_stat = getattr(np, stat)(redwine["alcohol"])
    print("Stat {} for alcohol content is: {}".format(stat, np_stat))
    return;

# Print KPI's for Alcohol content
print_stat_for_alcohol("max")
print_stat_for_alcohol("min")
print_stat_for_alcohol("mean")
print_stat_for_alcohol("median")

# #using MatPlotLib to visualise Data
# # using a simple linegraph to plot any correlation between pH and alcohol
x=(list(redwine["alcohol"]))
y=(list(redwine["pH"]))

plt.plot(x, y)

#using a scatter plot instead with labelled axis
plt.xlabel("Alcohol")
plt.ylabel("pH")
plt.scatter(x, y)
plt.show()

# Second Visual using Seaborn
datafra = pd.DataFrame(redwine.loc[0:100])
fig, ax = plt.subplots()
sns.countplot(x='alcohol', data=datafra).set_title('Count by Alcohol Content')
ax.set(xlabel="Alcohol Content", ylabel='Count')
plt.savefig('CountByAlcoholContent.jpg')
plt.show()
# High percentage within subset are withing 9.2 - 9.6 range in terms of alcohol content

# Third Visual
redwine = pd.read_csv("winequality-red.csv")
datafra = pd.DataFrame(redwine)
fig, ax = plt.subplots()
sns.countplot(x='quality', data=datafra).set_title('Count by Quality')
ax.set(xlabel="Quality", ylabel='Count')
plt.savefig('ByQuality.jpg')
plt.show()
# The majority of market is in middle - upper middle quality Wine. Does this also determine cost?
# If I was making a wine the best quality is not of the upmost importance, but it cannot be awful


# third Visual
fig, ax = plt.subplots()
sctplot = sns.scatterplot(y='density', x='alcohol', data=redwine, marker='o')
ax.set(xlabel="Alcohol Content", ylabel="Density of Wine")
sctplot.set_title('Density vs Alcohol')
plt.show()
plt.savefig('DensityVsAlcohol.jpg')
# As Density Decreases, in general so too does alcohol content

# fourth Visual
fig, ax = plt.subplots()
sctplot = sns.scatterplot(y='pH', x='fixed acidity', data=redwine, marker='o')
ax.set(xlabel="Acidity", ylabel = "pH level")
sctplot.set_title('pH Level vs Fixed Acidity')
plt.show()
plt.savefig('AcidVsPh.jpg')


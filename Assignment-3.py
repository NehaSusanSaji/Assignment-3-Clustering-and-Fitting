# -*- coding: utf-8 -*-
"""

@author: neham
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err

def read_data(file):
    """
    The function accepts a file and reads it into a pandas DataFrame and 
    cleans it and transposes it. It returns the cleaned original and 
    transposed DataFrame.

    Parameters
    ----------
    file : string
        The file name to be read into DataFrame.

    Returns
    -------
    df_clean : pandas DataFrame
        The cleaned version of the ingested DataFrame.
    df_t : pandas DataFrame
        The transposed version of the cleaned DataFrame.

    """
    
    # reads in an excel file
    if ".xlsx" in file:
        df = pd.read_excel(file, index_col=0)
    # reads in a csv file
    elif ".csv" in file:
        df = pd.read_csv(file, index_col=0)
    else:
        print("invalid filetype")
    # cleans the DataFrame
    df_clean = df.dropna(axis=1, how="all").dropna()
    # transposes the cleaned DataFrame
    df_t = df_clean.transpose()

    return df_clean, df_t

np.random.seed(10)
def kmeans_cluster(nclusters):
    kmeans = cluster.KMeans(n_clusters=nclusters)
    kmeans.fit(df_cluster)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    return labels, cen

def poly(x, a, b, c):
    
    x = x - 2003
    f = a + b*x + c*x**2
    
    return f


_, co2_df = read_data("co2_emissions.csv")
print(co2_df)

_, gdp_per_capita_df = read_data("gdp_per_capita.csv")
print(gdp_per_capita_df)

co2_india = co2_df.loc[:, "India"].copy()
print(co2_india)

gdp_per_capita_india = gdp_per_capita_df.loc["1990":"2019", "India"].copy()
print(gdp_per_capita_india)

df_india = pd.merge(co2_india, gdp_per_capita_india, on=co2_india.index,\
                    how="outer")
df_india = df_india.rename(columns={'key_0':"Year", 'India_x':"co2_emissions",\
                                    'India_y':"gdp_per_capita"})
df_india = df_india.set_index("Year")
print(df_india)

pd.plotting.scatter_matrix(df_india)


df_cluster = df_india[["co2_emissions", "gdp_per_capita"]].copy()

df_cluster, df_min, df_max = ct.scaler(df_cluster)

print("n   score")

for ncluster in range(2, 10):
    lab, cent = kmeans_cluster(ncluster) 
    print(ncluster, skmet.silhouette_score(df_cluster, lab))
    

label, center = kmeans_cluster(5)
xcen = center[:, 0]
ycen = center[:, 1]

plt.figure()
cm = plt.cm.get_cmap('Set1')
plt.scatter(df_cluster['gdp_per_capita'], df_cluster["co2_emissions"], s=10,\
            c=label, marker='o', cmap=cm)    
plt.scatter(xcen, ycen, s=20, c="k", marker="d")
plt.title("CO2 emission vs GDP per capita of India")
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions")
plt.show()

print(center)
centre = ct.backscale(center, df_min, df_max)
xcen = centre[:, 0]
ycen = centre[:, 1]
print(centre)


plt.figure()
cm = plt.cm.get_cmap('Set1')
plt.scatter(df_india['gdp_per_capita'], df_india["co2_emissions"], 10,
            label, marker='o', cmap=cm)    
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions")
plt.title("CO2 emission vs GDP per capita of India")
plt.show()

t = ['1990', '1995', '2000', '2005', '2010', '2015', '2020']

plt.plot(df_india.index, df_india['co2_emissions'])
plt.xlabel("Years")
plt.ylabel("CO2 Emissions (metric tons per capita)")
plt.title("CO2 Emissions (1990-2019)")
plt.xticks(ticks=t, labels=t)
plt.show()

plt.plot(df_india.index, df_india["gdp_per_capita"])
plt.xlabel("Years")
plt.ylabel("GDP per capita")
plt.title("GDP per capita (1990-2019)")
plt.xticks(ticks=t, labels=t)
plt.show()

df_india = df_india.reset_index()
df_india["gdp_per_capita"] = pd.to_numeric(df_india["gdp_per_capita"])
df_india["Year"] = pd.to_numeric(df_india["Year"])



param, covar = opt.curve_fit(poly, df_india["Year"],\
                             df_india["gdp_per_capita"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(1990, 2030)
forecast = poly(year, *param)
low, up = err.err_ranges(year, poly, param, sigma)
df_india["fit1"] = poly(df_india["Year"], *param)

plt.figure()
plt.plot(df_india["Year"], df_india["gdp_per_capita"], label="GDP", c='blue')
plt.plot(year, forecast, label="forecast", c='red')
plt.fill_between(year, low, up, color="yellow", alpha=0.8)
plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.title("GDP forecast of India")
plt.legend()
plt.show()


param, covar = opt.curve_fit(poly, df_india["Year"], df_india["co2_emissions"])
sigma = np.sqrt(np.diag(covar))
forecast = poly(year, *param)
low, up = err.err_ranges(year, poly, param, sigma)
df_india["fit2"] = poly(df_india["Year"], *param)

plt.figure()
plt.plot(df_india["Year"], df_india["co2_emissions"], label="CO2 emissions",
         c='green')
plt.plot(year, forecast, label="forecast", c="red")
plt.fill_between(year, low, up, color="yellow", alpha=0.8)
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (metric tons per capita)")
plt.title("CO2 Emissions Forecast of India")
plt.legend()
plt.show()

                             














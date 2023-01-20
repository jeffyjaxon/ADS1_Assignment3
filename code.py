import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def read_data(data_name):
    """
    Function reads data according to the file name passed and returns
    a dataframe

    """
    df = pd.read_csv(data_name, header=2)
    data = df.set_index('Country Name')
    data_final = data.drop(
        columns=['Country Code', 'Indicator Name', 'Indicator Code'])

    return data_final


def norm(array):
    """
    Function normalises an array of values

    """
    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array-min_val) / (max_val-min_val)

    return scaled


def norm_df(df, first=0, last=None):
    """
    Function normalises a dataframe by internally calling "norm" function and
    returns the normalised dataframe

    """
    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])

    return df


def heat_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns 
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


def expoFunc(x, a, b):
    """
    Function calculates the exponential function

    """
    return a**(x+b)


# Range of year
years = [str(num) for num in list(range(1990, 2020))]

# Calling read_data function to read csv file data
co2_emission = read_data("CO2_Emissions.csv")
population_growth = read_data("Population_Growth.csv")
agricultural_land = read_data("Agricultural_Land.csv")
gdp = read_data("GDP.csv")
arable_land = read_data("Arable_Land.csv")

# Filtering based on year range
agr_df = agricultural_land.loc[:, agricultural_land.columns.isin(years)]
pop_df = population_growth.loc[:, population_growth.columns.isin(years)]
co2_df = co2_emission.loc[:, co2_emission.columns.isin(years)]
gdp_df = gdp.loc[:, co2_emission.columns.isin(years)]
arable_df = arable_land.loc[:, arable_land.columns.isin(years)]

# Filtering value for first 40 countries
agr_df_final = agr_df.head(40)
pop_df_final = pop_df.head(40)
co2_df_final = co2_df.head(40)
gdp_df_final = gdp_df.head(40)
arable_df_final = arable_df.head(40)

final_df = pd.DataFrame()

# Retrieving value for the year 2014
final_df['Population'] = pop_df_final['2014']
final_df['CO2 Emission'] = co2_df_final['2014']
final_df['Agriculture land'] = agr_df_final['2014']
final_df['GDP'] = gdp_df_final['2014']
final_df['Arable Land'] = arable_df_final['2014']

# Replacing nan value with 0
final_df.replace(np.nan, 0, inplace=True)

# Calling function to generate heatmap
heat_corr(final_df, 9)

# Plotting scatter plots for the indicators
pd.plotting.scatter_matrix(final_df, figsize=(9.0, 9.0))
plt.tight_layout()
plt.show()


df_fitting = final_df[["Population", "Agriculture land"]].copy()

df_fitting = norm_df(df_fitting)

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fitting)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fitting, labels))

# Since silhouette score is highest for 3 , clustering for number = 3
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fitting)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Adding column with cluster information
cluster_df = final_df
cluster_df['Cluster'] = labels

plt.figure(figsize=(9.0, 9.0))
# Individual colours can be assigned to symbols. The label l is used to the select the
# l-th number from the colour table.
plt.scatter(df_fitting["Population"], df_fitting["Agriculture land"],
            c=labels, cmap="Accent", )

# Plotting cluster centre for 3 clusters
for ic in range(3):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc, "dk", markersize=10)


plt.xlabel("Population", fontsize = 15)
plt.ylabel("Agriculture land", fontsize = 15)
plt.title("Cluster Diagram with 3 clusters", fontsize = 15)
plt.show()

# Retrieving values for cluster 0, 1, 2
cluster_zero = pd.DataFrame()
cluster_one = pd.DataFrame()
cluster_two = pd.DataFrame()

# Selecting Population data for fitting
cluster_zero['Population'] = cluster_df[cluster_df['Cluster']
                                        == 0]['Population']
cluster_one['Population'] = cluster_df[cluster_df['Cluster'] == 1]['Population']
cluster_two['Population'] = cluster_df[cluster_df['Cluster'] == 2]['Population']

# Curve Fitting
plt.scatter(df_fitting["Population"], df_fitting["Agriculture land"])
popt, pcov = opt.curve_fit(
    expoFunc, df_fitting["Population"], df_fitting["Agriculture land"], p0=[-1, 1])


a_opt, b_opt = popt
x_mod = np.linspace(min(df_fitting["Population"]), max(
    df_fitting["Population"]), 100)
y_mod = expoFunc(x_mod, a_opt, b_opt)
print(y_mod)
plt.scatter(df_fitting["Population"], df_fitting["Agriculture land"])
plt.plot(x_mod, y_mod, color='r')
plt.title('Scatter plot after curve fitting')
plt.ylabel('Consumtion')
plt.xlabel('Access to electricity')
plt.show()

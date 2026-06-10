import pandas as pd

df = pd.read_csv('G:\\FluorescentCollagen\\20260302_ows2_col\\results_test_masked_min30\\ctfire_stats_per_slice.csv')


# Mean of all slices per stack + hist_type
#mean_all_slices = (
 #   df.groupby(['stack', 'hist_type'])['mean']
  #  .mean()
   # .reset_index()
    #.rename(columns={'mean': 'mean_of_slices'})
    #.pivot(index='stack',columns='hist_type', values = 'mean_of_slices')
#)

# Random slice per stack + hist_type
#random_slice = (
 #   df.groupby(['stack', 'hist_type'])
  #  .apply(lambda x: x.sample(1), include_groups=False)
   # .reset_index(drop=True)
#)

#print("Mean across all slices:")
#print(mean_all_slices)

#print("\nRandom slice selection:")
#print(random_slice)



# extract fields
df["concentration"] = df["stack"].str.extract(r'flu_(.*?)_roi')
df["roi"] = df["stack"].str.extract(r'_(roi\d+)')

# group and average
result = (
    df.groupby(["concentration", "roi", "hist_type"])["mean"]
    .mean()
    .unstack()
)

print(result)
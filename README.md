# High Frequency Trading Data pipeline
This repo will contain some helper functions in pandas/numpy/PyTorch which one uses quite often especially when working with High Frequency trading limit order books. Since the data is proprietary not everything will be shared.
Few words about the data we are working with and our goals

It's a crosssectional L2 level Limit order book data of National Stock Exchange spread over 10 months, over 66 days in 2024-2025, with over 15 million entries.
For a given stock at a given time we have: Top 10 bid/ask prices along with bid/ask sizes with same data recorded for past 100 events (which differ in micro to milliseconds).
These events are of 4 types: New, Trade, Cancel and Modify. We also record their side (Buy or Sell) along with the quantity and price. For the Modify event we also record the old price.
Our goal is to predict the mid price movement after 10 seconds. We already have recorded this data in a separate file.

We first need to clean the data, then we normalise to make our quantitites unitless as we are dealing with different stocks here. 
Then we engineer some new features and do z-score normalization across all channels before we apply our learning models which range from good old Linear Regression to Neural networks.


Our raw data is in csv files which we first convert to parquet to deal with the size issues. Since our access to GPU was limited we had to be selective to prevent overloading the gpu.
One important thing to keep in mind is to avoid for loops as much possible and vectorise whenever possible.
After that we remove all the rows which have nan/+-inf/0 values in bid/ask price and bid/ask size columns.
We also need to make 'windows' of 100 rows for each prediction (train or test). For that we have a column called first_tp. All rows in a given window share the same value in this column.
We make use of .groupby command in Pandas to count the number of rows in each window and drop the whole chunk of incomplete window and in case a window has more than 100 rows we truncate it over the latest 100 events.

Next step is alignment we the files where we have stored the prediction data which is stored in files with suffix fwdsampler.
Each row of fwd sampler has a column called time. the value stored in this column is also the same value stored in the time column of the 100th row in the corresponding window.
If for a given row in fwd sampler we don't find a row with the corresponding timestamp we remove that row and if for a given window we don't find a time stamp matching the timestamp of the 100th row 
in the window we drop the entire window.

Next step is normalization as different stocks trade at different range of values. We normalize ask/bid price as (ask/bid price - mid_100)/mid_100 where mid_100 = ask_price0+bid_price0/2.
100 here refers to the last row in the window.
We normalize ask/bid sizes by dividing by total ask+bd volume at last row of the window.
Now we add some features. Neural networks find it hard to learn multiplicative functions so we add columns ask/bid_pricei*bid/ask_sizei. Note that we cross multiply.
We need not add order book imbalance since given our normalization scheme it's a linear combination of features we already have. Think about it :)
We also add columns to take care of tick events.
At this point we split our data into training and test. We will consider last 6 days for the test.
Then we do z-normalization of train and test data by train mean and sd. it's crucial to not use test mean or sd to prevent leakage.

Now the data is ready for our models starting with linear regression. Our custom nn involved bilinear layers and multi headed attention layers, however i'm unable to share the precise architecture for propreitary reasons. The point of Bilinear layers is to learn the interaction across time and attention layers (with masking on teh time axis) learn the relative strength of the interactions. 




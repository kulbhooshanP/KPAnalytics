df = pd.read_csv('pizza.csv')

#Need to parse dates? Just pass in the corresponding column name(s).
df = pd.read_csv('pizza.csv', parse_dates=['dates'])

#Only need a few specific columns?
df = pd.read_csv('pizza.csv', usecols=['foo', 'bar'])
#The first thing you probably want to do is see what the data looks like. Here a few ways to check out Pandas data.

df.head()       # first five rows
df.tail()       # last five rows
df.sample(5)    # random sample of rows
df.shape        # number of rows/columns in a tuple
df.describe()   # calculates measures of central tendency
df.info()       # memory footprint and datatypes

#The quick and easy way is to just define a new column on the dataframe. This will give us column with the number 23 on every row. Usually, you will be setting the new column with an array or Series that matches the number of rows in the data.
df['new_column'] = 23
#Need to build a new column based on values from other columns?
full_price = (df.price + df.discount)
df['original_price'] = full_price
#Need the column in a certain order? The first argument is the position of the column. This will put the column at the begining of the DataFrame.
df.insert(0, 'original_price', full_price)

#Typically, I use .ix because it allows a mix of integers and strings. Enter the index of the row first, then the column.

df.ix[2, 'topping']
#You can also select the column first with dot notation, then the row index, which looks a little cleaner.

df.topping.ix[2]
#Either method will return the value of the cell.


#Let’s the we need to analyze orders that have pineapple in the topping column.

filtered_data = df[df.topping == 'pineapple']
#Or that meet a certain price threshold

filtered_data = df[df.price > 11.99 ]
#How about both at the same time? Just add the conditions to tuples and connect them with a bitwise operator.

filtered_data = df[(df.price > 11.99) & (df.topping == 'Pineapple')]

#Pretty self-explanatory, but very useful.

df.sort_values('price', axis=0, ascending=False)


#Anonymous lambda functions in Python are useful for these tasks. 
#Let’s say we need to calculate taxes for every row in the DataFrame with a custom function. 
#The pandas apply method allows us to pass a function that will run on every value in a column. 
#In this example, we extract a new taxes feature by running a custom function on the price data.

def calculate_taxes(price):
    taxes = price * 0.12
    return taxes

df['taxes'] = df.price.apply(calculate_taxes)
#Add a New Column with Conditional Logic
df['profitable'] = np.where(df['price']>=15.00, True, False)

#inding the Mean or Standard Deviation of Multiple Columns or Rows
#
If you have a DataFrame with the same type of data in every column, possibly a time series with financial data, you may need to find he mean horizontally.

df['mean'] = df.mean(axis=1)
#or to find the standard deviation vertically

df.std(axis=0)



#Converting a DataFrame to a Numpy Array
#Converting the the values in a DataFrame to an array is simple

df.values
#If you want to preserve the table presentation

df.as_matrix
#Combining DataFrames with Concatenation
#You can concatenate rows or columns together, the only requirement is that the shape is the same on corresponding axis. To concat rows vertically:

pd.concat([df_1, df_2], axis=0)
#Or to concat columns horizontally:

pd.concat([df_1, df_2], axis=1)
#Combining DataFrames based on an Index Key
#Merging in Pandas works just like SQL. If you you have two DataFrames that share a key, perhaps a pizza ‘order_id’, you can perform inner, outer, left, right joins just like you would in SQL.

merged_df = df_1.merge(df_2, how='left', on='order_id')
#Converting Dates to their own Day, Week, Month, Year Columns
#First, make sure the data is in datetime format. Then use dt method to extract the data you need.

date = pd.to_datetime(df.date)
df['weekday'] = date.dt.weekday
df['year'] = date.dt.year
#Finding NaNs in a DataFrame
#Count the total number of NaNs present:
df.isnull().sum().sum()
#List the NaN count for each column:
df.isnull().sum()
#Filling NaNs or Missing Data
#Most machine learning algorithms do not like NaN values, so you’ll probably need to convert them. If the topping column is missing some values, we can fill them a default value.
df.topping = df.topping.fillna('Cheese')
#or we can drop any row missing data across the entire DataFrame:
df = df.dropna(axis=0)
#Extracting Features by Grouping Columns
#Grouping columns is a great way to extract features from data. This is especially useful when you have data that can be counted or quantified in some way. For example, you might have group pizzas by topping, then calculate the mean for price in each group.
df.groupby('topping')['discount'].apply(lambda x: np.mean(x))
#or maybe you want to see the count of a certain value
df.groupby('topping')['discount'].apply(lambda x: x.count())

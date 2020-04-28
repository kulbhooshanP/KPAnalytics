df = pd.read_csv('pizza.csv')

#Need to parse dates? Just pass in the corresponding column name(s).
df = pd.read_csv('pizza.csv', parse_dates=['dates'])

#Only need a few specific columns?
df = pd.read_csv('pizza.csv', usecols=['foo', 'bar'])

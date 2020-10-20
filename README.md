# Task5(inputs)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
pwd
'C:\\Users\\Jyotismita Nandy'
Ship Mode	Segment	Country	City	State	Postal Code	Region	Category	Sub-Category	Sales	Quantity	Discount	Profit
0	Second Class	Consumer	United States	Henderson	Kentucky	42420	South	Furniture	Bookcases	261.9600	2	0.00	41.9136
1	Second Class	Consumer	United States	Henderson	Kentucky	42420	South	Furniture	Chairs	731.9400	3	0.00	219.5820
2	Second Class	Corporate	United States	Los Angeles	California	90036	West	Office Supplies	Labels	14.6200	2	0.00	6.8714
3	Standard Class	Consumer	United States	Fort Lauderdale	Florida	33311	South	Furniture	Tables	957.5775	5	0.45	-383.0310
4	Standard Class	Consumer	United States	Fort Lauderdale	Florida	33311	South	Office Supplies	Storage	22.3680	2	0.20	2.5164
5	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Furniture	Furnishings	48.8600	7	0.00	14.1694
6	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Office Supplies	Art	7.2800	4	0.00	1.9656
7	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Technology	Phones	907.1520	6	0.20	90.7152
8	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Office Supplies	Binders	18.5040	3	0.20	5.7825
9	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Office Supplies	Appliances	114.9000	5	0.00	34.4700
10	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Furniture	Tables	1706.1840	9	0.20	85.3092
11	Standard Class	Consumer	United States	Los Angeles	California	90032	West	Technology	Phones	911.4240	4	0.20	68.3568
12	Standard Class	Consumer	United States	Concord	North Carolina	28027	South	Office Supplies	Paper	15.5520	3	0.20	5.4432
13	Standard Class	Consumer	United States	Seattle	Washington	98103	West	Office Supplies	Binders	407.9760	3	0.20	132.5922
14	Standard Class	Home Office	United States	Fort Worth	Texas	76106	Central	Office Supplies	Appliances	68.8100	5	0.80	-123.8580
15	Standard Class	Home Office	United States	Fort Worth	Texas	76106	Central	Office Supplies	Binders	2.5440	3	0.80	-3.8160
16	Standard Class	Consumer	United States	Madison	Wisconsin	53711	Central	Office Supplies	Storage	665.8800	6	0.00	13.3176
17	Second Class	Consumer	United States	West Jordan	Utah	84084	West	Office Supplies	Storage	55.5000	2	0.00	9.9900
18	Second Class	Consumer	United States	San Francisco	California	94109	West	Office Supplies	Art	8.5600	2	0.00	2.4824
19	Second Class	Consumer	United States	San Francisco	California	94109	West	Technology	Phones	213.4800	3	0.20	16.0110
data.columns
data.columns
Index(['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Postal Code',
       'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount',
       'Profit'],
      dtype='object')
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9994 entries, 0 to 9993
Data columns (total 13 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Ship Mode     9994 non-null   object 
 1   Segment       9994 non-null   object 
 2   Country       9994 non-null   object 
 3   City          9994 non-null   object 
 4   State         9994 non-null   object 
 5   Postal Code   9994 non-null   int64  
 6   Region        9994 non-null   object 
 7   Category      9994 non-null   object 
 8   Sub-Category  9994 non-null   object 
 9   Sales         9994 non-null   float64
 10  Quantity      9994 non-null   int64  
 11  Discount      9994 non-null   float64
 12  Profit        9994 non-null   float64
dtypes: float64(3), int64(2), object(8)
memory usage: 702.8+ KB
data.isnull().sum()
data.isnull().sum()
Ship Mode       0
Segment         0
Country         0
City            0
State           0
Postal Code     0
Region          0
Category        0
Sub-Category    0
Sales           0
Quantity        0
Discount        0
Profit          0
dtype: int64
data.describe()
data.describe()
Postal Code	Sales	Quantity	Discount	Profit
count	9994.000000	9994.000000	9994.000000	9994.000000	9994.000000
mean	55190.379428	229.858001	3.789574	0.156203	28.656896
std	32063.693350	623.245101	2.225110	0.206452	234.260108
min	1040.000000	0.444000	1.000000	0.000000	-6599.978000
25%	23223.000000	17.280000	2.000000	0.000000	1.728750
50%	56430.500000	54.490000	3.000000	0.200000	8.666500
75%	90008.000000	209.940000	5.000000	0.200000	29.364000
max	99301.000000	22638.480000	14.000000	0.800000	8399.976000
data.corr()
Postal Code	Sales	Quantity	Discount	Profit
Postal Code	1.000000	-0.023854	0.012761	0.058443	-0.029961
Sales	-0.023854	1.000000	0.200795	-0.028190	0.479064
Quantity	0.012761	0.200795	1.000000	0.008623	0.066253
Discount	0.058443	-0.028190	0.008623	1.000000	-0.219487
Profit	-0.029961	0.479064	0.066253	-0.219487	1.000000
data['Category'].unique()
array(['Furniture', 'Office Supplies', 'Technology'], dtype=object)
data['Category'].value_counts()
Office Supplies    6026
Furniture          2121
Technology         1847
Name: Category, dtype: int64
data['Sub-Category'].nunique()
17
data['Sub-Category'].value_counts()
Binders        1523
Paper          1370
Furnishings     957
Phones          889
Storage         846
Art             796
Accessories     775
Chairs          617
Appliances      466
Labels          364
Tables          319
Envelopes       254
Bookcases       228
Fasteners       217
Supplies        190
Machines        115
Copiers          68
Name: Sub-Category, dtype: int64
plt.figure(figsize = (16,8))
plt.bar('Sub-Category', 'Category', data = data, color = 'y')
plt.show()

plt.figure(figsize = (12,10))
data['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()

data.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar
plt.title('Total Profit and Sales per Sub-Category')
plt.show()
<ipython-input-23-0c94a4116f37>:1: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
  data.groupby('Sub-Category')['Profit','Sales'].agg(['sum']).plot.bar

plt.figure(figsize = (15,8))
sns.countplot(x = 'Sub-Category', hue = 'Region', data = data)
plt.show()

plt.figure(figsize = (12,8))
sns.heatmap(data.corr(),annot = True,cmap = 'Set1')
<AxesSubplot:>

data['Cost'] = data['Sales']-data['Profit']
data['Cost'].head()
0     220.0464
1     512.3580
2       7.7486
3    1340.6085
4      19.8516
Name: Cost, dtype: float64
data['Ship Mode'].value_counts()
Standard Class    5968
Second Class      1945
First Class       1538
Same Day           543
Name: Ship Mode, dtype: int64
sortedTop20 = data.sort_values(['Profit'],ascending = False).head(20)
fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(111)
p = sns.barplot(x ='Segment',y = 'Profit',hue = 'State',palette = 'Set1' ,data = sortedTop20, ax = ax)
ax.set_title("Top 20 profitable customers")
ax.set_xtickables(p.get_xtickables(),rotation = 75)
plt.tight_layout()
plt.show()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-33-0d0a69342cb9> in <module>
      4 p = sns.barplot(x ='Segment',y = 'Profit',hue = 'State',palette = 'Set1' ,data = sortedTop20, ax = ax)
      5 ax.set_title("Top 20 profitable customers")
----> 6 ax.set_xtickables(p.get_xtickables(),rotation = 75)
      7 plt.tight_layout()
      8 plt.show()

AttributeError: 'AxesSubplot' object has no attribute 'set_xtickables'


plt.figure(figsize = (25,15))
sns.countplot(data['Ship Mode'],palette = 'Set3')
plt.title("Superstore's best shopping mode")
plt.show()
c:\users\jyotismita nandy\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(

plt.figure(figsize = (20,10))
sns.countplot('Ship Mode',data = data, hue = 'Segment', palette = 'rocket')
plt.show()
c:\users\jyotismita nandy\appdata\local\programs\python\python38-32\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(

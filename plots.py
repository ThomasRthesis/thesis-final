import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pandas as pd
from pandas.tools.plotting import table
import numpy as np

# Loading data
data = pd.read_pickle('data/complete.pkl')

# Opening file to dump tables and text
# Overwrite each time when script is run
fs = open('plots/stats.txt', 'w')

# Making sure the data are integers
list_num = ['age', 'id', 'followers_count', 'friends_count']
data[list_num] = data[list_num].apply(pd.to_numeric, errors='coerce')

# Splitting data
df_num = data[['age', 'followers_count', 'friends_count']]
df_text = data[['bot', 'face', 'gender', 'signal']]

# Simple statistics and ratios
total_size = data.shape[0]
fs.write('Total size dataframe: ' + str(total_size) + '\n\n')

genders = data['gender'].value_counts()
m = genders['m']
f = genders['f']
o = genders['o']
b = genders['-']
total = m+f+o+b

fs.write('Total amount of m: ' + str(m) + ' , ' + str(m/total_size) + '\n')
fs.write('Total amount of f: ' + str(f) + ' , ' + str(f/total_size) + '\n')
fs.write('Total amount of o: ' + str(o) + ' , ' + str(o/total_size) + '\n')
fs.write('Total amount of b: ' + str(b) + ' , ' + str(b/total_size) + '\n\n')

faces = data['gender'][(data['face']) & (data['gender'] != '-')].value_counts()
faces_m = faces['m']
faces_f = faces['f']
faces_o = faces['o']
faces_total = faces.sum()

fs.write('Total m with pp: ' + str(faces_m) + ' , ' + str(faces_m/faces_total) + '\n')
fs.write('Total f with pp: ' + str(faces_f) + ' , ' + str(faces_f/faces_total) + '\n')
fs.write('Total o with pp: ' + str(faces_o) + ' , ' + str(faces_o/faces_total) + '\n')
fs.write('Total with pp: ' + str(faces_total) + ' , ' + str(faces_total/total) + '\n\n')

fs.write('% of m with pp: ' + str(faces_m/m) + '\n')
fs.write('% of f with pp: ' + str(faces_f/f) + '\n')
fs.write('% of o with pp: ' + str(faces_o/o) + '\n\n')

no_faces = data['gender'][(data['face']==False) & (data['gender'] != '-')].value_counts()
no_faces_m = no_faces['m']
no_faces_f = no_faces['f']
no_faces_o = no_faces['o']
no_faces_total = no_faces.sum()

df_faces = pd.DataFrame([
    [faces_m, faces_m/faces_total, no_faces_m, no_faces_m/no_faces_total],
    [faces_f, faces_f/faces_total, no_faces_f, no_faces_f/no_faces_total],
    [faces_o, faces_o/faces_total, no_faces_o, no_faces_o/no_faces_total]],
                        columns=['faces', 'faces%', 'no_faces', 'no_faces%'],
                        index=['m','f','o'])

fs.write('Total amount of m without pp: ' + str(no_faces_m) + ' , ' + str(no_faces_m/no_faces_total) + '\n')
fs.write('Total amount of f without pp: ' + str(no_faces_f) + ' , ' + str(no_faces_f/no_faces_total) + '\n')
fs.write('Total amount of o without pp: ' + str(no_faces_o) + ' , ' + str(no_faces_o/no_faces_total) + '\n')
fs.write('Total amount without pp: ' + str(no_faces_total) + '\n\n')

bots = data[data['bot']]['bot'].sum()
fs.write('Number of bots/pages: ' + str(bots) + '\n\n')

# Summaries
summary_num = df_num.describe().transpose().round(2)
summary_text = df_text.describe(include='all').transpose().round(2)
fs.write('summary numerical variables \n')
fs.write(str(summary_num) + '\n\n')
fs.write('summary text variables \n')
fs.write(str(summary_text) + '\n\n')


# Percentiles
followers = data['followers_count']
quantiles = [x/10 for x in range(11)]
bins = followers.quantile(q=quantiles)
bins = list(bins)
y = pd.cut(followers, bins, labels=False).as_matrix()
fs.write('percentiles:' + str(quantiles) + '\n')
fs.write('bins:' + str(bins) + '\n')
fs.write('bincount:' + str(np.bincount(followers)) + '\n\n')

quantiles = [x/20 for x in range(0, 21, 5)]
bins = followers.quantile(q=quantiles)
bins = list(bins)
y = pd.cut(followers, bins, labels=False).as_matrix()
fs.write('percentiles:' + str(quantiles) + '\n')
fs.write('bins:' + str(bins) + '\n')
fs.write('bincount:' + str(np.bincount(followers)) + '\n\n')

# Signal Counts
def signal_counter(df, gender):
    df = df[df['gender']==gender]
    df = df['signal'].apply(lambda x: pd.value_counts(x.split(' '))).sum(axis=0)
    # remove filled nans and empty ones
    df.drop(['rm', ''], inplace=True)
    return df

signal = data.copy()

# fill nans to prevent error later on when removing
signal.fillna('rm', inplace=True)

# No subset
signal_all = signal['signal'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
signal_all.drop(['rm',''], inplace=True)

# Combined WITH and WITHOUT profile image
signal_m = signal_counter(signal, 'm')
signal_f = signal_counter(signal, 'f')
signal_o = signal_counter(signal, 'o')
signal_b = signal_counter(signal, '-')

# Subset signal WITH profile picture
# Bots/pages have no profile picture
signal_pp = data.copy()
signal_pp = signal_pp[signal_pp['face']]
signal_pp.fillna('rm', inplace=True)

signal_m_pp = signal_counter(signal_pp, 'm')
signal_f_pp = signal_counter(signal_pp, 'f')
signal_o_pp = signal_counter(signal_pp, 'o')

# Subset signal WITHOUT profile image
signal_np = data.copy()
signal_np = signal_np[signal_np['face']==False]
signal_np.fillna('rm', inplace=True)

signal_m_np = signal_counter(signal_np, 'm')
signal_f_np = signal_counter(signal_np, 'f')
signal_o_np = signal_counter(signal_np, 'o')
signal_b_np = signal_counter(signal_np, '-')

fs.write('signal all: \n' + str(signal_all) + '\n')
fs.write('signal m all: \n' + str(signal_m) + '\n')
fs.write('signal f all: \n' + str(signal_f) + '\n')
fs.write('signal o all: \n' + str(signal_o) + '\n')
fs.write('signal b all: \n' + str(signal_b) + '\n\n')

fs.write('signal m pp: \n' + str(signal_m_pp) + '\n')
fs.write('signal f pp: \n' + str(signal_f_pp) + '\n')
fs.write('signal o pp: \n' + str(signal_o_pp) + '\n')

fs.write('signal m np: \n' + str(signal_m_np) + '\n')
fs.write('signal f np: \n' + str(signal_f_np) + '\n')
fs.write('signal o np: \n' + str(signal_o_np) + '\n')
fs.write('signal b np: \n' + str(signal_b_np) + '\n\n')

## Plots
plot_count = 0

# Saving them in their own folder named 'plots'
# Save fig as x-y-f.png with 3 letters per variable
def factorize_data(column):
    column = pd.factorize(column)
    labels = column[1]
    column = pd.value_counts(column[0])
    x = column.keys()
    y = column
    return x, y, labels

##
fig = plt.figure()
fig.suptitle('Age~Followers -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['age']
y = subset['followers_count']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)
coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_xlabel('Age')
ax.set_ylabel('Followers')

plt.savefig('plots/age-fol-gen.png')
print('plot age followers gender done')
plot_count+=1

##
fig = plt.figure()
#fig.suptitle('Age~Followers(l - 5000) -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['age']
y = subset['followers_count']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)
coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_ylim([0, 5000])
ax.set_xlabel('Age')
ax.set_ylabel('Followers')

plt.savefig('plots/age-fol(l5000)-gen.png')
print('plot age followers gender done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Age~Followers(l) -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['age']
y = subset['followers_count']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)
coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_ylim([0, 20000])
ax.set_xlabel('Age')
ax.set_ylabel('Followers')

plt.savefig('plots/age-fol(l)-gen.png')
print('plot age followers gender done')
plot_count+=1

##
fig = plt.figure()
#fig.suptitle('Following~Followers(l) -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['friends_count']
y = subset['followers_count']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)
coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_ylim([0, 20000])
ax.set_xlim([0, 10000])
ax.set_xlabel('Following')
ax.set_ylabel('Followers')

plt.savefig('plots/fri-fol(l)-gen.png')
print('plot friends followers gender done')
plot_count+=1


fig = plt.figure()
#fig.suptitle('Following~Followers(l) -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['friends_count']
y = subset['followers_count']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)
coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_ylim([0, 5000])
ax.set_xlim([0, 5000])
ax.set_xlabel('Following')
ax.set_ylabel('Followers')

plt.savefig('plots/fri(l)-fol(l)-gen.png')
print('plot friends followers gender done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Following~Followers -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['friends_count']
y = subset['followers_count']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)
coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_xlabel('Following')
ax.set_ylabel('Followers')

plt.savefig('plots/fri-fol-gen.png')
print('plot friends followers gender done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Age~Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset[['age', 'gender']]
x = [
    data[data['gender']=='m']['age'],
    data[data['gender']=='f']['age'],
    data[data['gender']=='o']['age']
    ]

print('min age:', data['age'].min())
print('max age:', data['age'].max())
bin_min = float(data['age'].min())
bin_max = float(data['age'].max())
bins = np.arange(bin_min, bin_max)
ax.hist(x=x, range=(bin_min, bin_max), bins=bins, stacked=True, label=['Male', 'Female', 'Other'])

ax.legend(prop={'size': 10})
ax.set_xlabel('Age')
ax.set_ylabel('Count')

plt.savefig('plots/age-gen.png')
print('plot age gender done')
plot_count+=1

##
fig = plt.figure()
#fig.suptitle('Followers~Age -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['followers_count']
y = subset['age']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)

# linear regression line
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)
ax.set_xlabel('Followers')
ax.set_ylabel('Age')

plt.savefig('plots/fol-age-gen.png')
print('plot followers age gender done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Followers(l)~Age -- Gender', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(axis=0, subset=['age'], inplace=True)
x = subset['followers_count']
y = subset['age']
fill_c = pd.factorize(subset['gender'])[0]

ax.scatter(x=x, y=y,
           marker='+', s=100, linewidths=2, c=fill_c, cmap=plt.cm.coolwarm, alpha=0.3)

coeffs = np.polyfit(x, y, 3, rcond=None, full=False, w=None, cov=False)
x2 = np.arange(min(x)-1, max(x)+1, .01)
y2 = np.polyval(coeffs, x2)
plt.plot(x2, y2)

ax.set_xlim([0, 20000])
ax.set_xlabel('Followers')
ax.set_ylabel('Age')

plt.savefig('plots/fol(l)-age-gen.png')
print('plot followers(l) age gender done')
plot_count+=1

##
x, y, labels = factorize_data(data['gender'])

fig = plt.figure()
fig.suptitle('Distribution Gender/Bot', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

ax.bar(x, y, width=1)
ax.set_xticks([0.5, 1.5, 2.5, 3.5])
ax.set_xticklabels(labels)

plt.savefig('plots/gen-gen.png')
print('plot gender distribution done')
plot_count+=1

##
x2, y2, labels2 = factorize_data(data['gender'][data['face']])
fig = plt.figure()
fig.suptitle('Distribution Gender/Bot with face', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

ax.bar(x, y, width=1)
ax.bar(x2, y2, width=1, color='red')
ax.set_xticks([0.5, 1.5, 2.5, 3.5])
ax.set_xticklabels(labels)

plt.savefig('plots/gen-gen-fac.png')
print('plot gender distribution with face done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Spread Age', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset.dropna(subset=['age'], inplace=True)
subset = [
    subset[subset['gender']=='m']['age'],
    subset[subset['gender']=='f']['age'],
    subset[subset['gender']=='o']['age']
]

plt.boxplot(subset)

ax.set_xlabel('Gender')
ax.set_ylabel('Age')

plt.savefig('plots/gen-age-boxplot.png')
print('boxplot gen age done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Spread Followers', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset = [
    subset[subset['gender']=='m']['followers_count'],
    subset[subset['gender']=='f']['followers_count'],
    subset[subset['gender']=='o']['followers_count'],
    subset[subset['gender']=='-']['followers_count']
]

plt.boxplot(subset)

ax.set_xlabel('Gender')
ax.set_ylabel('Followers')

plt.savefig('plots/gen-fol-boxplot.png')
print('boxplot gen followers done')
plot_count+=1

##
fig = plt.figure()
fig.suptitle('Spread Followers(l)', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.9)

subset = data.copy()
subset = [
    subset[subset['gender']=='m']['followers_count'],
    subset[subset['gender']=='f']['followers_count'],
    subset[subset['gender']=='o']['followers_count'],
    subset[subset['gender']=='-']['followers_count']
]

plt.boxplot(subset)

ax.set_ylim([0, 20000])
ax.set_xlabel('Gender')
ax.set_ylabel('Followers')

plt.savefig('plots/gen-fol(l)-boxplot.png')
print('boxplot gen followers done')
plot_count+=1

##
fig = plt.figure()
#fig.suptitle('Spread Followers(l)', fontsize=15, fontweight='bold')
ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.9)

subset = data.copy()
subset = [
    subset[subset['gender']=='m']['followers_count'],
    subset[subset['gender']=='f']['followers_count'],
    subset[subset['gender']=='o']['followers_count'],
    subset[subset['gender']=='-']['followers_count']
]

plt.boxplot(subset)

ax.set_ylim([0, 5000])
ax.set_xlabel('Gender')
ax.set_xticklabels(['Male', 'Female', 'Other', 'Bot/Page'])
ax.set_ylabel('Followers')

plt.savefig('plots/gen-fol(l5000)-boxplot.png')
print('boxplot gen followers done')
plot_count+=1

##
N = len(signal_m)
ind = np.arange(N)
width = 0.20
labels = signal_m.index

url_series=pd.Series([0], index=['url'])
signal_o = signal_o.append(url_series)

signal_m = signal_m.values
signal_f = signal_f.values
signal_o = signal_o.values
signal_b = signal_b.values

signal_m = signal_m/sum(signal_m)
signal_f = signal_f/sum(signal_f)
signal_o = signal_o/sum(signal_o)
signal_b = signal_b/sum(signal_b)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, signal_m, width, color='b')
rects2 = ax.bar(ind + width, signal_f, width, color='y')
rects3 = ax.bar(ind + width*2, signal_o, width, color='g')
rects4 = ax.bar(ind + width*3, signal_b, width, color='r')

ax.set_ylabel('Counts (%)')
ax.set_title('Signals')
ax.set_xticks( ind + width * 2)
ax.set_xticklabels(labels)

ax.legend((rects1[0], rects2[0], rects3[0], rects4), ('Male', 'Female', 'Other', 'Bot/Page'))

plt.savefig('plots/sig-sig-all.png')
print('plot signals all done')
plot_count+=1

##
N = len(signal_m_pp)
ind = np.arange(N)
width = 0.30
labels = signal_m_pp.index

url_series=pd.Series([0], index=['url'])
signal_o_pp = signal_o_pp.append(url_series)

signal_m = signal_m_pp.values
signal_f = signal_f_pp.values
signal_o = signal_o_pp.values

signal_m = signal_m/sum(signal_m)
signal_f = signal_f/sum(signal_f)
signal_o = signal_o/sum(signal_o)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, signal_m, width, color='b')
rects2 = ax.bar(ind + width, signal_f, width, color='y')
rects3 = ax.bar(ind + width*2, signal_o, width, color='g')

ax.set_ylabel('Counts (%)')
#ax.set_title('Signal profile images')
ax.set_xticks( ind + width * 2 - (width/2))
ax.set_xticklabels(labels)

ax.legend((rects1[0], rects2[0], rects3[0], rects4), ('Male', 'Female', 'Other'))

plt.savefig('plots/sig-sig-pp.png')
print('plot signals pp done')
plot_count+=1

##
N = len(signal_m_np)
ind = np.arange(N)
width = 0.20
labels = signal_m_np.index

url_series=pd.Series([0], index=['url'])
signal_o_np = signal_o_np.append(url_series)
signal_o_np = signal_o_np.append(pd.Series([0], index=['image']))
signal_f_np = signal_f_np.append(url_series)

signal_m = signal_m_np.values
signal_f = signal_f_np.values
signal_o = signal_o_np.values
signal_b = signal_b_np.values

signal_m = signal_m/sum(signal_m)
signal_f = signal_f/sum(signal_f)
signal_o = signal_o/sum(signal_o)
signal_b = signal_b/sum(signal_b)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, signal_m, width, color='b')
rects2 = ax.bar(ind + width, signal_f, width, color='y')
rects3 = ax.bar(ind + width*2, signal_o, width, color='g')
rects4 = ax.bar(ind + width*3, signal_b, width, color='r')

ax.set_ylabel('Counts (%)')
#ax.set_title('Signal no profile images')
ax.set_xticks( ind + width * 2 - (width/2))
ax.set_xticklabels(labels)

ax.legend((rects1[0], rects2[0], rects3[0], rects4), ('Male', 'Female', 'Other', 'Bot/Page'))

plt.savefig('plots/sig-sig-np.png')
print('plot signals np done')
plot_count+=1

#
fig = plt.figure()
ax = fig.add_subplot(111)

subset = data.copy()
x = subset['friends_count'].as_matrix()

plt.hist(x, 10000, normed=1)

ax.set_xlabel('Followers')
ax.set_ylabel('Count')

plt.savefig('plots/fol-fol.png')
print('plot followers count done')
plot_count+=1

#
fig = plt.figure()
ax = fig.add_subplot(111)

subset = data.copy()
x = subset['friends_count'].as_matrix()

plt.hist(x, 10000, normed=1)

ax.set_xlabel('Followers')
ax.set_xlim([0, 20000])
ax.set_ylabel('Count')

plt.savefig('plots/fol(l)-fol.png')
print('plot followers count done')
plot_count+=1

#
fig = plt.figure()
ax = fig.add_subplot(111)

subset = data.copy()
x = subset['friends_count'].as_matrix()

plt.hist(x, 10000, normed=1)

ax.set_xlabel('Followers')
ax.set_xlim([0, 5000])
ax.set_ylabel('Count')

plt.savefig('plots/fol(l2)-fol.png')
print('plot followers count done')
plot_count+=1

print('plot count: ', plot_count)
print('stats.py done')


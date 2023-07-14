import numpy as np
import openpyxl as openpyxl
import xlrd
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from math import log10
from matplotlib import cm
import seaborn as sns
from math import pi
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

filename = 'chip_dataset cpus vs gpus.csv'
filename2='Intel_CPUs.csv'
filename3='gpu.csv'
filename4='All_GPUs.csv'

info = pd.read_csv(filename)
df=pd.read_csv(filename2)
gpu=pd.read_csv(filename3)
AllG=pd.read_csv(filename4)

dimensions1 = info.shape
dim2=df.shape
dim3=gpu.shape
dim4=AllG.shape

columns1=dimensions1[1]
rows1=dimensions1[0]
column_header1=info.columns.tolist()

columns2=dim2[1]
rows2=dim2[0]
column_header2=df.columns.tolist()

columns3=dim3[1]
rows3=dim3[0]
column_header3=gpu.columns.tolist()

columns4=dim4[1]
rows4=dim4[0]
column_header4=AllG.columns.tolist()

print(dimensions1)
print(columns1)
print(rows1)
print(column_header1)

print(dim2)
print(columns2)
print(rows2)
print(column_header2)

print(dim3)
print(columns3)
print(rows3)
print(column_header3)

print(dim4)
print(columns4)
print(rows4)
print(column_header4)

## GPU percentages that support DX 12.0 seperated by Vendor// pie plot// 'All_GPUs.csv'

manufacturer_counts = AllG['Manufacturer'].value_counts()
directX12=AllG[AllG['Direct_X'] == 'DX 12.0']['Manufacturer'].value_counts()
manufacturer_percentages = (directX12 / directX12.sum()) * 100

dx12_percentages = (directX12 / manufacturer_counts) * 100

print(dx12_percentages)


fig, ax = plt.subplots(figsize=(6, 6))
ax = plt.subplot(projection='polar')
startangle = 90
colors = ['#4393E5', '#43BAE5', '#7AE6EA','#6e6ee8']
xs = [(i * pi * 2) / 100 for i in dx12_percentages]
ys = [-0.2, 1, 2.2,3.4]
left = (startangle * pi * 2) / 360
for i, x in enumerate(xs):
    ax.barh(ys[i], x, left=left, height=1, color=colors[i])
    ax.scatter(x + left, ys[i], s=350, color=colors[i], zorder=2)
    ax.scatter(left, ys[i], s=350, color=colors[i], zorder=2)

plt.ylim(-4, 4)
legend_elements = [Line2D([0], [0], marker='o', color='w', label='AMD', markerfacecolor='#4393E5', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='ATI', markerfacecolor='#43BAE5', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Intel', markerfacecolor='#7AE6EA', markersize=10)
                   ,Line2D([0], [0], marker='o', color='w', label='Nvidia', markerfacecolor='#6e6ee8', markersize=10)]
ax.legend(handles=legend_elements, loc='center', frameon=False)
plt.xticks([])
plt.yticks([])
plt.title('percentages of GPUs that support DX 12.0 per manufacturer')
ax.spines.clear()
plt.show()

## plot of all inte CPUs temperatures//hexagonal bin// 'Intel_CPUs.csv'

valid_temperatures = []

for temperature in df['T']:
    if pd.isnull(temperature) or not temperature.endswith('°C') or not temperature[:-2].isnumeric():
        print("Invalid or missing data detected.")
        continue
    else:
        stripped_temperature = float(temperature[:-2])
        valid_temperatures.append(stripped_temperature)

print(valid_temperatures)

valid_df = pd.DataFrame({'Temperature': valid_temperatures})
hexbin_plot = plt.hexbin(valid_df.index, valid_df['Temperature'], gridsize=15, cmap='YlOrRd', bins='log')

plt.xlabel('Index')
plt.ylabel('Temperature (°C)')
plt.title('Intel CPUs Temperatures ')

cbar = plt.colorbar(hexbin_plot)
cbar.set_label('Count')
plt.show()

##Bus Type and GPU clock frequency // count of each clock frequency // 1-bar plot//2-histogram// 'gpu.csv'

gpu['clock_num'] = ''
for index, row in gpu.iterrows():
    original_data = row['MemoryClock']
    if original_data=='System Shared':
        extracted_data=800
    else:
        extracted_data=int(original_data.split()[0].strip())

    gpu.at[index, 'clock_num'] = extracted_data

memBus=gpu.groupby('Bus')['clock_num'].mean()

sns.set_theme()
gradient_color = sns.color_palette("Blues")[2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.bar(memBus.index, memBus.values, color=gradient_color)
ax1.set_xlabel('Bus Type')
ax1.set_ylabel('Memory Clock freq in Hz')
ax1.set_title('Distribution of Memory Clock')
ax1.set_xticklabels(memBus.index, rotation=45, ha='right')

ax2.hist(gpu['clock_num'], bins=10, edgecolor='black', color=gradient_color)
ax2.set_xlabel('Memory Clock frequencies in Hz')
ax2.set_ylabel('count')
ax2.set_title('Count of Memory Clock')

for rect in ax1.patches:
    rect.set_color(gradient_color)
    rect.set_edgecolor('black')
    rect.set_linewidth(1)
    rect.set_hatch('////')
    rect.set_alpha(0.8)

for rect in ax2.patches:
    rect.set_color(gradient_color)
    rect.set_edgecolor('black')
    rect.set_linewidth(1)
    rect.set_hatch('////')
    rect.set_alpha(0.8)

plt.tight_layout()
plt.show()

## comparing process averages of different manufacturers/vendors using different files to check for data deviation // bar plot// 'All_GPUs.csv'


AllG['Process_Num'] = AllG['Process'].str.extract(r'(\d+)').astype(float)
process_avg = AllG.groupby('Manufacturer')['Process_Num'].mean()
process_array = np.array(process_avg)

gpu_data = info[info['Type'] == 'GPU']
avg_process_per_vendor = gpu_data[gpu_data['Vendor'] != 'Other'].groupby('Vendor')['Process Size (nm)'].mean()
result_array = np.array(avg_process_per_vendor)

combined_array = np.column_stack((process_avg, avg_process_per_vendor))
vendors = process_avg.index.tolist()
x = np.arange(len(vendors))

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.4
opacity = 0.8

bar1 = ax.bar(x - 0.2, process_array, width=bar_width, label='Vendor Average', alpha=opacity, color='blue')
bar2 = ax.bar(x + 0.2, result_array, width=bar_width, label='Manufacturer Average', alpha=opacity, color='green')

for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate('{}'.format(round(height, 1)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Vendors')
ax.set_ylabel('Average Process')
ax.set_title('Average Process Comparison between Vendors and Manufacturers', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(vendors)
ax.legend()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()



###Percentage of GPU production based on Vendor // Pie plot// 'chip_dataset cpus vs gpus.csv'

numGPU = info[info['Type'] == 'GPU'].groupby('Vendor')['Type'].count()
vendor_counts = numGPU.reset_index()
vendor_counts.columns = ['Vendor', 'GPU Count']

colors = sns.color_palette('Set3')

fig, ax = plt.subplots(figsize=(10, 6))
wedges, labels, percentages = plt.pie(
    vendor_counts['GPU Count'],
    labels=vendor_counts['Vendor'],
    autopct='%1.1f%%',
    colors=colors,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    startangle=90,
    textprops={'fontsize': 12}
)

sorted_labels = [label.get_text() for label in labels]
sorted_percentages = [f'{percent.get_text()}%' for percent in percentages]

plt.legend(wedges, sorted_labels, title='Vendor', loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Distribution of GPUs per Vendor', pad=20)

circle = plt.Circle((0, 0), 0.6, color='white')
plt.gca().add_artist(circle)

plt.axis('equal')
plt.show()


###the growth of transistors over the years // line plot // 'chip_dataset cpus vs gpus.csv'

info['Year'] = pd.to_datetime(info['Release Date']).dt.year
avgT = info.groupby('Year')['Transistors (million)'].mean()

plt.plot(avgT.index, avgT, marker='o', linestyle='-', color='pink')
plt.xlabel('Year')
plt.ylabel('Transistors (million)')
plt.title('Average Transistors per Year')

plt.xticks(rotation=90)

background_color = 'lightgray'
plt.fill_between(avgT.index, avgT, color=background_color)

plt.show()

### TDP comparison between vendors // 1- bar plot using 'chip_dataset cpus vs gpus.csv'// 2-bar plot using data from 'All_GPUs.csv'
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5))

avgTDP = info.groupby('Vendor')['TDP (W)'].mean()
a1.bar(avgTDP.index, avgTDP, color='pink')
a1.set_xlabel('Vendor')
a1.set_ylabel('TDP per Vendor (Watt)')
a1.set_title('Average TDP per Vendor')

AllG['Max_Power_Num'] = AllG['Max_Power'].str.extract(r'(\d+)').astype(float)
avgP = AllG.groupby('Manufacturer')['Max_Power_Num'].mean()
a2.bar(avgP.index, avgP, color='pink')
a2.set_xlabel('Manufacturer')
a2.set_ylabel('Max Power (Watt)')
a2.set_title('Maximum Power for GPUs by Vendor')
plt.tight_layout()
plt.show()


##Relationship between Lithography and proccessor speed // scatter plot //'Intel_CPUs.csv'

mode_lithography = df['Lithography'].value_counts().idxmax()

df['Lithography'].fillna(mode_lithography, inplace=True)
df[['Lithography_Num', 'Lithography_Unit']] = df['Lithography'].str.split(n=1, expand=True)

lithography_sizes = df['Lithography_Num'].tolist()

processor_frequencies1 = df['Processor_Base_Frequency'].astype(str).str.split(n=1, expand=True)[0].tolist()
processor_frequencies=[]
for i in processor_frequencies1:
    freq_value=float(i)
    ##in case it is in MHz
    if freq_value>5:
        freq_value=freq_value/1000
    processor_frequencies.append(freq_value)

sns.set_style("ticks")

sns.scatterplot(x=lithography_sizes, y=processor_frequencies, palette="husl")

sns.set_palette("plasma")

plt.rcParams['axes.facecolor'] = '#E6E6FA'
sns.violinplot(x=lithography_sizes, y=processor_frequencies, palette="plasma")

plt.xlabel('Lithography Size (nm)')
plt.ylabel('Processor Base Frequency (GHz)')
plt.title('Lithography Size vs. Processor Base Frequency')

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()




















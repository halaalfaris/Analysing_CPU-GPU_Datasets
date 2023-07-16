This repository contains Python code for analyzing and visualizing data related to CPUs and GPUs. The code uses various libraries such as NumPy, pandas, Matplotlib, seaborn, and openpyxl. Below is an overview of the functionality provided by each section of the code:

#Data Loading and Exploration The code begins by importing the necessary libraries and loading the data from multiple CSV files into pandas DataFrames. The files used are:

1-'chip_dataset cpus vs gpus.csv' 
2-'Intel_CPUs.csv' 
3-'gpu.csv' 
4-'All_GPUs.csv' 
The dimensions (number of rows and columns) of each DataFrame are calculated, and the column headers are extracted for further analysis.

#GPU Percentages Supporting DX 12.0 (All_GPUs.csv) This section calculates the percentage of GPUs that support DX 12.0 per manufacturer using the 'All_GPUs.csv' data. It creates a pie plot to visualize the distribution of these percentages among different manufacturers.

#Intel CPUs Temperatures (Intel_CPUs.csv) The code extracts valid temperature data from the 'Intel_CPUs.csv' file and creates a hexagonal bin plot to visualize the distribution of temperatures across Intel CPUs.

#Bus Type and GPU Clock Frequency (gpu.csv) This section processes the 'gpu.csv' data to extract clock frequencies and calculates the average clock frequency for each bus type. It creates two plots: a bar plot showing the distribution of memory clock frequencies by bus type and a histogram showing the count of memory clock frequencies.

#Process Averages Comparison (All_GPUs.csv) The code calculates the average process size for each manufacturer from the 'All_GPUs.csv' data. It also calculates the average process size per vendor using the 'chip_dataset cpus vs gpus.csv' data. It creates a bar plot to compare the process averages between vendors and manufacturers.

#Percentage of GPU Production by Vendor (chip_dataset cpus vs gpus.csv) This section calculates the percentage of GPU production for each vendor using the 'chip_dataset cpus vs gpus.csv' data. It creates a pie plot to visualize the distribution of GPU production among different vendors.

#Transistors Growth over the Years (chip_dataset cpus vs gpus.csv) The code calculates the average number of transistors per year using the 'chip_dataset cpus vs gpus.csv' data. It creates a line plot to visualize the growth of transistors over the years.

#TDP Comparison between Vendors This section calculates the average TDP (Thermal Design Power) per vendor using the 'chip_dataset cpus vs gpus.csv' data. It creates a bar plot to compare the TDP values between vendors. Additionally, it calculates the maximum power for GPUs by manufacturer using the 'All_GPUs.csv' data and creates another bar plot to visualize this information.

#Relationship between Lithography and Processor Speed (Intel_CPUs.csv) The code analyzes the relationship between lithography size and processor base frequency using the 'Intel_CPUs.csv' data. It creates a scatter plot and a violin plot to visualize this relationship.

Please note that this code assumes the availability of the specified CSV files and the necessary dependencies. Ensure that the files are in the correct format and located in the same directory as the code before running it.

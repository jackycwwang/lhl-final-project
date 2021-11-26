# Lighthouse Labs Final Project

This is the back-to-back capstone project in Lighthouse Labs Data Science bootcamp after 10-week intensive DS training. 

![](images/lhl_logo.jpg)

*- Source: [Lighthouse Labs](https://www.lighthouselabs.ca/)*

## Topic: Predictive/Preventive Maintenance

### Introduction and Motivation

Machine failure is unpleasant and unproductive. The production line or business operation has to halt days or weeks  until the machine is back to normal. This can incur a huge lost to businesses.

Predictive maintenance is ML/AI-based program that is capable of monitoring the behavior of a machine, or machine component, predicting or "foreseeing" machine failure, and thus allowing the opportunities to take preventative measures.

Such preventative actions can bring substantial benefits to businesses whose business operation relies on expensive machine/device, such as manufacturing, telecoms, healcare systems, etc. Furthermose, the use case of this technique don't just limit to detecting machine failure. There are many business cases that are similar to this problem domain, which can be generalized to detect anomaly events, such as fraud detection, image denoising, cypersecurity, etc.
![lhl_logo](https://user-images.githubusercontent.com/1385633/143627797-6437fef6-978f-471d-9f08-b2689dde58a2.jpg)


![](images/predictive_maintenance.jpg)

*- Source: [PCI](https://www.pcimag.com/articles/106046-predictive-maintenance-and-its-role-in-improving-efficiency)*


### The Challenges
There are quite a few challenges, which can make failure prediction inaccurate, as outlined below:
1. In most cases, the dataset is extremely imbalanced since anomaly usually are rare events. This makes ML/AI struggle to make the classifications.
2. Oversampling or undersampling techniques may not be helpful.
3. The distribution of the normal case may overlap with the distribution of the anomaly case, which makes the seperation difficult resulting in low performance in modeling.
4. The feature space may be small, which may hinder the ML/AI model to find accurate underlying patterns. 

### Dataset:

This dataset is taken from [UCI-Machine Learning Depository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset#)

The original credit goes to : Stephan Matzka, School of Engineering - Technology and Life, Hochschule fÃ¼r Technik und Wirtschaft Berlin, 12459 Berlin, Germany, stephan.matzka'@'htw-berlin.de

#### This dataset is a synthesized dataset that mimics the real predictive maintenance data that gathers from certain device, as quotes here:
> *Since real predictive maintenance datasets are generally difficult to obtain and in particular difficult to publish, we present and provide a synthetic dataset that reflects real predictive maintenance encountered in industry to the best of our knowledge.*

#### Here are the detailed dataset description :
The dataset consists of 10 000 data points stored as rows with 14 features in columns

- **UID**: unique identifier ranging from 1 to 10000
- **Product ID**: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
- **Air temperature [K]**: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
- **Process temperature [K]**: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- **Rotational speed [rpm]**: calculated from a power of 2860 W, overlaid with a normally distributed noise
- **Torque [Nm]**: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values.
- **Tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. and a
- **'Machine failure'**: label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true.

#### The machine failure consists of five independent failure modes

- tool wear failure (**TWF**): the tool will be replaced of fail at a randomly selected tool wear time between 200 â€“ 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
- heat dissipation failure (**HDF**): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the toolâ€™s rotational speed is below 1380 rpm. This is the case for 115 data points.
- power failure (**PWF**): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
- overstrain failure (**OSF**): if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
- random failures (**RNF**): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.

(Note: This project does not take into the consideration to these different failure modes)

If at least one of the above failure modes is true, the process fails and the `Machine failure` label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail.


### Approaches:
Since the dataset in this domain usually consists of mostly numerical values, the feature engineering is usually not so required, and the preprocessing steps are also minimal. We sould only use one-hot encoding for categories and scaling techniques for numerics.

We explored this dataset in three different approaches: 
- Classic Machine Learning
- Artificial Neural Network/Deep Learning
- Autoencoders

And we followed the following procedure:
1. Exploratory Data Analysis
2. Build a baseline model - Logistic Regression
3. Build a Machine Learning model - Random Forest
4. Build a Deep Learning model - ANN 
5. Build an Autoencoder model - Undercomplete model

### Findings:
- The failure percentage is about 3.4%
![](images/failure_percentage.jpg)

### Folder and Files:


### Tools used:


### What can go Further:



### Reference:
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.




#!/usr/bin/env python
# coding: utf-8

# # **Predicting Family Planning Demand and Optimizing Service Delivery in Kenya**

# ## Project Overview
# This project aims to enhance family planning (FP) service delivery efficiency by leveraging data-driven approaches, specifically machine learning (ML) using the CRISP-DM methodology. The goal is to improve maternal and child health outcomes, reduce unmet need for family planning, and contribute to achieving national and global development goals related to reproductive health.
# 
# ## Problem Statement
# Kenya faces significant strides in increasing access to family planning services, yet a substantial unmet need for family planning remains. According to the 2022 Kenya Demographic and Health Survey (KDHS), the total unmet need for family planning is 15%, with 10% for spacing and 5% for limiting births. This indicates that a significant portion of the population desires to space or limit births but is not using any contraceptive method. Traditional methods often show imbalances, with a heavy reliance on short-acting methods, leading to unstable uptake of long-acting reversible contraceptives (LARCs) and permanent methods. This disparity can lead to higher discontinuation rates and continued unmet need. Supply chain inefficiencies, commodity stock-outs, inadequate healthcare worker training, and uneven distribution of resources exacerbate these issues, hindering effective service delivery.
# 
# ## Stakeholders
# * **Government of Kenya:** Ministry of Health (MoH), especially the National Family Planning Coordinated Implementation Plan (NFPICIP) and Kenya Health Information System (KHIS) initiatives.
# * **Policymakers and Donors:** Need evidence-based advocacy for resource mobilization and investment.
# * **Healthcare Providers:** Frontline health workers providing FP services.
# * **Women of Reproductive Age:** Direct beneficiaries of improved FP services.
# * **Local Communities:** Impacted by and involved in FP service delivery.
# 
# ## Key Statistics
# * **Total Unmet Need for Family Planning (2022 KDHS):** 15%
#     * **Unmet Need for Spacing:** 10%
#     * **Unmet Need for Limiting Births:** 5%
# * **Married Women Aged 15-49 Rising to 57% in 2022 (DRS 2022):**  Modern contraceptive prevalence rate (mCPR).
# * **Number of new clients for FP method band and the continuation rate:** Key data points for predicting future demand.
# 
# ## Key Analytics Questions
# * How many new clients are expected for injectables in a County next quarter?
# * What is the projected demand for different family planning methods (injectables, pills, condoms, implants, IUD, sterilization) at various geographical (e.g., county) and temporal (e.g., quarterly, annual) granularities?
# * How can we optimize resource allocation (commodities, equipment, staffing) to meet projected demand and minimize wastage?
# * How can predictive analytics identify potential stock-outs or oversupply of specific FP commodities in different locations?
# * Which regions or demographics are most underserved in terms of family planning access and uptake?
# 
# ## Objectives
# * **Quantitatively forecast the demand for specific family planning methods:** This includes predicting the continuation rates of users for each method at defined geographical and temporal scales.
# * **Enable proactive resource allocation:** This involves optimizing the distribution of commodities, equipment, and staffing to reduce stock-outs, minimize wastage, and improve targeted interventions.
# * **Improve method continuation:** By understanding demand and improving service delivery, the project aims to reduce discontinuation rates and increase sustained use of FP methods.
# * **Provide evidence-based insights:** Support policymakers and donors in making informed decisions regarding resource mobilization and investment in family planning.
# 
# ## Metrics of Importance to Focus On
# * **Accuracy of Demand Forecasts:** Measured by comparing predicted demand with actual uptake for various FP methods at different geographical and temporal levels (e.g., Mean Absolute Error, Root Mean Squared Error).
# * **Commodity Stock-out Rates:** Reduction in the number or duration of stock-outs for essential family planning commodities.
# * **Resource Utilization Efficiency:** Metrics related to optimal allocation and reduced wastage of commodities, equipment, and human resources.
# * **Method Continuation Rates:** Increase in the percentage of users who continue using a specific family planning method over a defined period (e.g., 12-month continuation rate).
# * **Unmet Need for Family Planning:** Contribution to the reduction of the national unmet need for family planning.
# * **Client Satisfaction:** Indirectly improved through better access and availability of preferred methods.
# * **Healthcare Worker Productivity:** Optimized allocation of staff to meet demand efficiently.

# # Preliminaries

# ## Importing Python Libraries

# In[198]:


#Import necessary libraries for data manipulation, visualization, and machine learning
# Data handling and manipulation
import pandas as pd                  # For dataframes and data manipulation
import numpy as np                   # For numerical operations and arrays

# Visualization libraries
import matplotlib.pyplot as plt      # For basic plotting
import seaborn as sns                # For advanced statistical visualizations

# Suppress warning messages
import warnings                      # To filter out warnings during execution

# Model selection and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
# train_test_split: splits data into training and test sets
# cross_val_score: performs cross-validation
# TimeSeriesSplit: cross-validation for time series data such as this FP data
from sklearn.model_selection import GridSearchCV

# Data preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
# StandardScaler/MinMaxScaler: scale numerical features
# OneHotEncoder: encode categorical variables

# Metrics for model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# MAE, MSE, R²: performance metrics for regression models

# Combine preprocessing steps
from sklearn.compose import ColumnTransformer  # Apply different preprocessing to columns

# Build machine learning workflows
from sklearn.pipeline import Pipeline          # Chain preprocessing and modeling steps

# Handle missing data
from sklearn.impute import SimpleImputer       # Fill missing values

# Regression models
from sklearn.linear_model import LinearRegression, Ridge                   # Linear regression model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


# RandomForestRegressor: ensemble method using decision trees
# GradientBoostingRegressor: boosting method for better accuracy


# ## Data Loading

# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')  # Mount Google Drive


# base_path = "/content/drive/MyDrive/data/"

# # File paths
# population_data_path = base_path + "ke_fp_population_data.csv"
# service_data_path = base_path + "ke_fp_service_data.csv"
# benchmarks_data_path = base_path + "ke_fp_benchmarks_data.csv"
# commodity_data_path = base_path + "ke_fp_commodity_data.csv"

# # Attempt to read with 'latin1' encoding
# df_population = pd.read_csv(population_data_path, encoding='latin1')
# df_service = pd.read_csv(service_data_path, encoding='latin1')
# df_benchmarks = pd.read_csv(benchmarks_data_path, encoding='latin1')
# df_commodity = pd.read_csv(commodity_data_path, encoding='latin1')


# In[3]:


# Data loading

try:
    # Define paths
    data_dir = "data/"
    benchmarks_dir = "data/benchmarks/"

    population_data_path = f"{data_dir}ke_fp_population_data.csv"
    service_data_path = f"{data_dir}ke_fp_service_data.csv"
    commodity_data_path = f"{data_dir}ke_fp_commodity_data.csv"

    benchmarks_core_health_workforce_data_path = f"{benchmarks_dir}ke_fp_benchmarks_core_health_workforce.csv"
    benchmarks_demand_satisfied_data_path = f"{benchmarks_dir}ke_fp_benchmarks_Demand_Satisfied.csv"
    benchmarks_mCPR_data_path = f"{benchmarks_dir}ke_fp_benchmarks_mCPR.csv"
    benchmarks_teen_pregnancy_data_path = f"{benchmarks_dir}ke_fp_benchmarks_Teenage_Pregnancy_rate.csv"
    benchmarks_unmet_need_data_path = f"{benchmarks_dir}ke_fp_benchmarks_Total_Unmet_Need_MW.csv"

    # Load CSVs with fallback encoding
    df_population = pd.read_csv(population_data_path, encoding='latin1')
    df_service = pd.read_csv(service_data_path, encoding='latin1')
    df_commodity = pd.read_csv(commodity_data_path, encoding='latin1')

    # Load benchmark-specific CSVs
    df_core_health_workforce = pd.read_csv(benchmarks_core_health_workforce_data_path, encoding='latin1')
    df_demand_satisfied = pd.read_csv(benchmarks_demand_satisfied_data_path, encoding='latin1')
    df_mcpr = pd.read_csv(benchmarks_mCPR_data_path, encoding='latin1')
    df_teenage_pregnancy = pd.read_csv(benchmarks_teen_pregnancy_data_path, encoding='latin1')
    df_unmet_need = pd.read_csv(benchmarks_unmet_need_data_path, encoding='latin1')

    # Success logs
    print("Datasets loaded successfully:")
    print(f"{population_data_path} shape: {df_population.shape}")
    print(f"{service_data_path} shape: {df_service.shape}")
    print(f"{commodity_data_path} shape: {df_commodity.shape}")
    print(f"{benchmarks_core_health_workforce_data_path} shape: {df_core_health_workforce.shape}")
    print(f"{benchmarks_demand_satisfied_data_path} shape: {df_demand_satisfied.shape}")
    print(f"{benchmarks_mCPR_data_path} shape: {df_mcpr.shape}")
    print(f"{benchmarks_teen_pregnancy_data_path} shape: {df_teenage_pregnancy.shape}")
    print(f"{benchmarks_unmet_need_data_path} shape: {df_unmet_need.shape}")

except FileNotFoundError as e:
    print("❌ Error: One or more CSV files were not found.")
    print("🔍 Please ensure all expected files are in their respective folders.")
    print(e)
except Exception as e:
    print("❗ An unexpected error occurred while loading the datasets:")
    print(e)


# # **1. Data Understanding**

# ## *a) Data Cleaning*
# 
# This involved;
# * Standardization of the column names
# * Renaming the columns
# * Dropping empty and unwanted columns
# * Handling missing values, duplicates and outliers

# ### 1. ke_fp_service_data.csv

# In[4]:


# Make a copy of the data
df_service1=df_service.copy()


# In[5]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[6]:


# Preview the data
df_service1.head()


# In[7]:


# Explore the data
df_service1.info()


# In[8]:


# Standardize the column names

def standardize_col_labels(df):
    def clean_column(col):
        # Remove redundant prefixes
        col = col.replace('MOH 711 Rev ', '')
        col = col.replace('MOH 711 ', '')

        # Formatting
        col = col.strip().lower()          # Convert to lowercase
        col = col.replace(' ', '_')     # Replace spaces with underscores
        col = col.replace('-', '_')  # Replace hyphen with underscores
        col = col.replace('â€“', '_')
        col = col.replace('â', '_')
        return col

    df.columns = [clean_column(col) for col in df.columns]
    return df

df_service1 = standardize_col_labels(df_service1)
df_service1.columns


# In[9]:


# Rename column names

name_map = {
    'periodcode': 'year_month',
    'orgunitlevel1': 'country',
    'orgunitlevel2': 'county',
    'organisationunitid': 'uid',
    'organisationunitcode':'county_code',
    'county_cou':'county_code',
    'dataname':'population_indicator'
}
df_service1 = df_service1.rename(columns=name_map)
df_service1


# In[10]:


# Drop columns where all values are null
df_service1=df_service1.dropna(axis=1, how='all')
df_service1


# In[11]:


# Drop unwanted columns
df_service1=df_service1.drop(columns=['periodid','organisationunitname', 'periodname','population_growth_rate',
                                       'total_population','women_of_childbearing_age_(15_49yrs)'
                                       ], axis=1)


# In[12]:


# Check for missing values
df_service1.isna().sum().sort_values(ascending=False)


# Missing values were interpreted as 'no service was provided or dataset missing for the organization unit' and filled with 0

# In[13]:


# Fill the missing values with zeros
df_service1 = df_service1.fillna(0)


# In[14]:


# Extract year from 'year_month'
df_service1['year'] = df_service1['year_month'].astype(str).str[:4].astype(int)
df_service1


# In[15]:


# Create a new column (uid_code)
df_service1['uid_code'] = df_service1[
    ['year_month','uid' ]].astype(str).agg('_'.join, axis=1)

df_service1.head()


# In[16]:


# Create a new column (uid_year)
df_service1['uid_year'] = df_service1[
    ['year','uid' ]].astype(str).agg('_'.join, axis=1)

df_service1.head()


# In[17]:


df_service1.columns


# In[18]:


# Check columns with float dtypes
float_cols = df_service1.select_dtypes(include=['float', 'float64']).columns
float_cols


# In[19]:


# Convert float columns to int64
df_service1[float_cols] = df_service1[float_cols].astype('int64')


# In[20]:


df_service1.info()


# ### 2. ke_fp_commodity_data.csv

# In[21]:


# Make a copy of the data
df_commodity1 = df_commodity.copy()


# In[22]:


# Preview the data
df_commodity1.head()


# In[23]:


# Explore the data
df_commodity1.info()


# In[24]:


# Standardize the column names
df_commodity1 = standardize_col_labels(df_commodity1)

df_commodity1.head()


# In[25]:


# Rename column names
df_commodity1 = df_commodity1.rename(columns=name_map)
df_commodity1


# In[26]:


# Drop columns where all values are null
df_commodity1=df_commodity1.dropna(axis=1, how='all')
df_commodity1


# In[27]:


# Drop unwanted columns
df_commodity1.drop(columns='organisationunitname', inplace=True)
df_commodity1


# In[28]:


# Check for missing values
df_commodity1.isna().sum().sort_values(ascending=False)


# The missing values were interpreted as 'no event reported'

# In[29]:


# Dealing with the missing values

df_commodity1 = df_commodity1.fillna(0)


# In[30]:


# Check for duplicates
df_commodity1.duplicated().sum()


# A new column(uid_code) was created by concatenating the year_month column and organisation unit id

# In[31]:


# Create a new column (uid_code)
df_commodity1['uid_code'] = df_commodity1[['year_month','uid' ]].astype(str).agg('_'.join, axis=1)

df_commodity1.head()


# In[32]:


# Convert float columns to int64
float_cols_com = df_commodity1.select_dtypes(include=['float', 'float64']).columns
df_commodity1[float_cols_com] = df_commodity1[float_cols_com].astype('int64')
df_commodity1.info()


# ### 3. ke_fp_population_data

# In[33]:


# Make a copy of the data
df_population1 =df_population.copy()


# In[34]:


# Standardize column names
df_population1 = standardize_col_labels(df_population1)

# Preview the data
df_population1.head()


# In[35]:


# Rename column names
df_population1 = df_population1.rename(columns=name_map)

df_population1.head()


# In[36]:


# Drop unwanted columns
df_population1.drop(columns=['organisationunitname'], axis=1) # This was dropped because it is the same as county


# #### Calculate Women eligible for FP

# In[37]:


# Insert the new column, treating NaN values as 0 during the calculation
df_population1['eligible_fp'] = (
    df_population1['women_of_childbearing_age_(15_49yrs)']
    .add(df_population1['population_10_14_year_old_girls'], fill_value=0)
    .sub(df_population1['estimated_number_of_pregnant_women'], fill_value=0)
)

# Display the updated DataFrame
df_population1.head()


# ### 4. ke_fp_benchmarks_data.csv

# In[38]:


# Make copies of the benchmarks dataframes
df_core_health_workforce1 = df_core_health_workforce.copy()
df_demand_satisfied1 = df_demand_satisfied.copy()
df_mcpr1 = df_mcpr.copy()
df_teenage_pregnancy1 = df_teenage_pregnancy.copy()
df_unmet_need1 = df_unmet_need.copy()


# In[39]:


# Preview the data
df_core_health_workforce1.head()
df_demand_satisfied1.head()
df_mcpr1.head()
df_teenage_pregnancy1.head()
df_unmet_need1.head()


# ## *b) Initial Feature Engineering*

# ### CYP computation and grouping

# Couple Years of Protection(CYP)-CYP measures the estimated protection provided by FP based on the volume of contraceptive method distribution to clients to help monitor health system performance and track trends and progress over time.

# In[40]:


# Define CYP conversion factors
cyp_factors = {
    'condoms': 0.0083,
    'emergency_pill': 0.05,
    'pills_combined_oral_contraceptives': 0.0067,
    'pills_progestin_only_contraceptives': 0.0833,
    'injections': 0.25,
    'implants_1_rod': 2.5,
    'implants_2_rod': 3.8,
    'iucd_hormonal': 4.8,
    'iucd_non_hormonal': 4.6,
    'surgical': 10.0
}

# Initialize total_cyp column
df_service1['total_cyp'] = 0

# Compute total_cyp based on matching column names
for col in df_service1.columns:
    if 'condom' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['condoms']

    elif 'emergency' in col and 'pill' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['emergency_pill']

    elif 'combined_oral' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['pills_combined_oral_contraceptives']

    elif 'progestin_only' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['pills_progestin_only_contraceptives']

    elif 'injection' in col or 'dmpa' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['injections']

    elif '1_rod' in col and 'implant' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['implants_1_rod']

    elif '2_rod' in col and 'implant' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['implants_2_rod']

    elif 'iucd' in col and 'hormonal' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['iucd_hormonal']

    elif 'iucd' in col and 'non_hormonal' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['iucd_non_hormonal']

    elif 'surgical' in col or 'vasectomy' in col or 'btl' in col:
        df_service1['total_cyp'] += df_service1[col] * cyp_factors['surgical']


# In[41]:


df_service1.head()


# ### FP Method Grouping (New vs Revisits)

# In[42]:


# Group FP Methods
df_service1['adolescent_10_24_receiving_fp_new'] = (
    df_service1['adolescent_10_14_yrs_receiving_fp_services_new_clients'] +
    df_service1['adolescent_15_19_yrs_receiving_fp_services_new_clients'] +
    df_service1['adolescent_20_24_yrs_receiving_fp_services_new_clients']
)

df_service1['adolescent_10_24_receiving_fp_revisits'] = (
    df_service1['adolescent_10_14_yrs_receiving_fp_services_re_visits'] +
    df_service1['adolescent_15_19_yrs_receiving_fp_services_re_visits'] +
    df_service1['adolescent_20_24_yrs_receiving_fp_services_re_visits']
)

df_service1['adults_25+_receiving_fp_services_new'] = df_service1['2020_adults_25+_receiving_fp_services_new_clients']
df_service1['adults_25+_receiving_fp_services_revisits'] = df_service1['2020_adults_25+_receiving_fp_services_re_visits']

df_service1['condoms_new'] = df_service1['clients_receiving_female_condoms_new_clients'] + df_service1['client_receiving_male_condoms_new_clients']
df_service1['condoms_revisits'] = (
    df_service1['clients_receiving_female_condoms_re_visits'] +
    df_service1['client_receiving_male_condoms_re_visits']
)

df_service1['pills_new'] = (
    df_service1['emergency_contraceptive_pill_new_clients'] +
    df_service1['pills_combined_oral_contraceptive_new_clients'] +
    df_service1['pills_progestin_only_new_clients']
)

df_service1['pills_revisits'] = (
    df_service1['emergency_contraceptive_pill_re_visits'] +
    df_service1['pills_combined_oral_contraceptive_re_visits'] +
    df_service1['pills_progestin_only_re_visits']
)

df_service1['injectable_new'] = (
    df_service1['2020_fp_injections_dmpa__im_new_clients'] +
    df_service1['2020_fp_injections_dmpa__sc_new_clients']
)

df_service1['injectable_revisits'] = (
    df_service1['2020_fp_injections_dmpa__im_re_visits'] +
    df_service1['2020_fp_injections_dmpa__sc_re_visits']
)

df_service1['implants_new'] = (
    df_service1['2020_implants_insertion_1_rod_ist_time_insertion'] +
    df_service1['2020_implants_insertion_2_rod_ist_time_insertion']
)

df_service1['implants_revisits'] = (
    df_service1['2020_implants_insertion_1_rod_re_insertion'] +
    df_service1['2020_implants_insertion_2_rod_re_insertion']
)

df_service1['iucd_new'] = (
    df_service1['2020_iucd_insertion_hormonal_ist_time_insertion'] +
    df_service1['2020_iucd_insertion_non_hormonal_ist_time_insertion']
)

df_service1['iucd_revisits'] = (
    df_service1['2020_iucd_insertion_hormonal_re_insertion'] +
    df_service1['2020_iucd_insertion_non_hormonal_re_insertion']
)

df_service1['surgical_new'] = (
    df_service1['2020_voluntary_surgical_contraception_vasectomy_ist_time_insertion'] +
    df_service1['2020_voluntary_surgical_contraception_btl_ist_time_insertion']
)

df_service1['surgical_revisits'] = (
    df_service1['2020_voluntary_surgical_contraception_vasectomy_re_insertion'] +
    df_service1['2020_voluntary_surgical_contraception_btl_re_insertion']
)

df_service1['traditional_new'] = (
    df_service1['2020_clients_given_cycle_beads_new_clients'] +
    df_service1['clients_counselled_natural_family_planning_new_clients']
)

df_service1['traditional_revisits'] = (
    df_service1['2020_clients_given_cycle_beads_re_visits'] +
    df_service1['clients_counselled_natural_family_planning_re_visits']
)


# In[43]:


df_service1


# ### FP Method overal banding (pills, condoms, injectables,implants,iucd & surgical)

# In[44]:


# Combined method categories
df_service1['condoms'] = df_service1['condoms_new'] + df_service1['condoms_revisits']
df_service1['pills'] = df_service1['pills_new'] + df_service1['pills_revisits']
df_service1['injectables'] = df_service1['injectable_new'] + df_service1['injectable_revisits']
df_service1['implants'] = df_service1['implants_new'] + df_service1['implants_revisits']
df_service1['iucd'] = df_service1['iucd_new'] + df_service1['iucd_revisits']
df_service1['surgical'] = df_service1['surgical_new'] + df_service1['surgical_revisits']


# Compute Total mmodern FP methods
df_service1['total_modern_fp'] = (
    df_service1['condoms'] +
    df_service1['pills'] +
    df_service1['injectables'] +
    df_service1['implants'] +
    df_service1['iucd'] +
    df_service1['surgical']
)


# Compute Total traditional FP methods
df_service1['traditional'] = df_service1['traditional_new'] + df_service1['traditional_revisits']


# In[45]:


# List of columns to keep
keep_cols = [
    'year_month', 'country', 'county', 'uid', 'uid_code','uid_year', 'county_code',
    'adolescent_10_24_receiving_fp_new',
    'adolescent_10_24_receiving_fp_revisits',
    'adults_25+_receiving_fp_services_new',
    'adults_25+_receiving_fp_services_revisits',
    'condoms_new', 'condoms_revisits', 'traditional_new',
    'pills_new', 'pills_revisits', 'traditional_revisits',
    'injectable_new', 'injectable_revisits', 'traditional',
    'implants_new', 'implants_revisits',
    'iucd_new', 'iucd_revisits',
    'surgical_new', 'surgical_revisits',
    'condoms', 'pills', 'injectables', 'implants', 'iucd', 'surgical',
    'total_modern_fp', 'total_cyp'
]

# Keep only the specified columns
df_service1 = df_service1[keep_cols]


# #### Drop underlying columns

# In[46]:


df_service1.columns


# ### Joining the datasets
# 
# A left join was used to merge the four datasets. The service data was used as the base data. Organisation unit id code (uid_code) was used to join the four datasets

# In[47]:


# Merge df_service1 with df_commodity1
fp_service_comm_df = pd.merge(df_service1, df_commodity1, how='left', on='uid_code')

fp_service_comm_df


# In[48]:


# Rename the column
fp_service_comm_df= fp_service_comm_df.rename(columns={"county_code_x": "county_code"})

fp_service_comm_df


# In[49]:


# Replace non-finite values (NaN, inf, -inf) with 0
fp_service_comm_df = fp_service_comm_df.replace([np.inf, -np.inf], np.nan).fillna(0)


# In[50]:


# Convert float columns to int64
df_float_cols = fp_service_comm_df.select_dtypes(include=['float', 'float64']).columns
df_float_cols


# In[51]:


fp_service_comm_df[df_float_cols] = fp_service_comm_df[df_float_cols].astype('int64')


# In[52]:


fp_service_comm_df.info()


# In[53]:


# Perform a left merge to retain all rows from fp_service_comm_df
fp_service_comm_pop_df = fp_service_comm_df.merge(
    df_population1[['uid_year', 'eligible_fp']], 
    on='uid_year', 
    how='left'
)

# Preview the merged DataFrame
fp_service_comm_pop_df.head()


# In[54]:


# Merge df_core_health_workforce
data_df = fp_service_comm_pop_df.merge(
    df_core_health_workforce[['uid_code', 'core_health_workforce_per_10,000population']],
    on='uid_code', how='left'
)


# In[55]:


# Merge df_demand_satisfied
data_df = data_df.merge(
    df_demand_satisfied1[['uid_code', 'Demand_Satisfied_by_Modern_Methods (%)']],
    on='uid_code', how='left'
)


# In[56]:


# Merge df_mcpr
data_df = data_df.merge(
    df_mcpr1[['uid_code', 'mCPR (Married Women, %)']],
    on='uid_code', how='left'
)


# In[57]:


# Merge df_teenage_pregnancy
data_df = data_df.merge(
    df_teenage_pregnancy1[['uid_code', 'Teenage Pregnancy Rate (15-19, %)']],
    on='uid_code', how='left'
)


# In[58]:


# Merge df_unmet_need
data_df = data_df.merge(
    df_unmet_need1[['uid_code', 'Total Unmet Need (Married Women, %)']],
    on='uid_code', how='left'
)


# In[59]:


# Rename the columns by removing suffixes
data_df.columns = [
    col.replace('_x_x', '')
       .replace('_y_y', '')
       .replace('_x', '')
       .replace('_y', '')
       for col in data_df.columns
]


# In[60]:


# Preview the data to make sure the columns have been renamed
data_df.head()


# In[61]:


data_df.shape


# In[62]:


# Remove duplicate columns
data_df = data_df.loc[:, ~data_df.columns.duplicated()]


# In[63]:


# Checking the merged data for duplicates
data_df.duplicated().sum()


# In[64]:


# Replace " county" label with blanks in county column

data_df['county'] = data_df['county'].str.replace(' County', '', regex=False)
print(data_df[['county']].head())


# ### Export .csv file for data_df

# In[65]:


#data_df_sorted = data_df.sort_values(by='year_month', ascending=False)

# Export to CSV
#data_df_sorted.to_csv("output/data_df.csv", index=False, encoding='utf-8')


# ### Composite Calculations

# In[170]:


# Calculate total users receiving FP services
data_df['total_users_receiving_fp'] = (
   data_df['adolescent_10_24_receiving_fp_new'] +
   data_df['adolescent_10_24_receiving_fp_revisits'] +
   data_df['adults_25+_receiving_fp_services_new'] +
   data_df['adults_25+_receiving_fp_services_revisits']
)

# Total actual new (Modern fp)
data_df['total_actual_new_modern_fp_methods'] = data_df[['pills_new', 'condoms_new', 
                                                         'injectable_new', 'implants_new', 
                                                         'iucd_new', 'surgical_new']].sum(axis=1)


# Total actual revisits(Modern fp)
data_df['total_actual_revisits_modern_fp_methods'] = data_df[['pills_revisits', 'condoms_revisits', 
                                                              'injectable_revisits', 
                                                              'implants_revisits', 'iucd_revisits', 
                                                              'surgical_revisits']].sum(axis=1)


# Total actual traditional fp
data_df['total_actual_traditional_methods'] = data_df[['traditional_revisits', 'traditional_revisits']].sum(axis=1)


# Total actual modern fp
data_df['total_actual_modern_fp'] = (
    data_df['total_actual_new_modern_fp_methods'] + 
    data_df['total_actual_revisits_modern_fp_methods']
)

# Total FP (both Modern and Traditional)
data_df['total_fp'] = (data_df['total_actual_modern_fp']  + 
                       data_df['total_actual_traditional_methods'])

# Proportion adolescents receiving fp
data_df['proportion_adolescents_10_24_yrs_receiving_fp'] = (( data_df['adolescent_10_24_receiving_fp_new'] +
                                                            data_df['adolescent_10_24_receiving_fp_revisits']) / 
                                                            data_df['total_users_receiving_fp'] )

# Number of months
data_df['number_of_months'] = min(data_df['year_month'].nunique(), 12)

# Actual mCPR
data_df['actual_mcpr'] = data_df['total_actual_modern_fp'] / data_df['eligible_fp']

# Actual mCPR monthly
data_df['actual_mcpr_monthly'] = data_df['total_modern_fp'] / (data_df['eligible_fp'] / 12)

# Number of women(10_49_yrs) with unmet need for fp
data_df['number_women_10_49_yrs_with_unmet_need_for_fp'] = data_df['eligible_fp'] - data_df['total_actual_modern_fp']

# Unmet need
data_df['actual_unmet_need_for_modern_fp'] = (
    data_df['number_women_10_49_yrs_with_unmet_need_for_fp'] / 
    data_df['eligible_fp']
)

# Actual core health workers-2013 
data_df['actual_core_health_workers_2013_10000_eligible_fp'] = (data_df['eligible_fp'] / 10000)

# Actual total demand
data_df['actual_total_demand_for_fp'] = data_df['actual_mcpr'] + data_df['actual_unmet_need_for_modern_fp']

# Actual demand satisfied
data_df['actual_demand_satisfied']  = data_df['actual_mcpr'] / data_df['actual_total_demand_for_fp']


# Combine pills (COCs + POCs)
data_df['total_pills_dispensed'] = (
    data_df['pills_combined_oral_contraceptive_stock_dispensed'] +
    data_df['pills_progestin_only_pills_stock_dispensed']
)

# Combine condoms (male + female)
data_df['total_condoms_dispensed'] = (
    data_df['condoms_male_condom_stock_dispensed'] +
    data_df['condoms_female_condom_stock_dispensed']
)

# Total all commodities dispensed
data_df['total_commodities_dispensed'] = (
    data_df['total_pills_dispensed'] +
    data_df['total_condoms_dispensed'] +
    data_df['injectables_stock_dispensed'] +
    data_df['implants_stock_dispensed'] +
    data_df['iud_stock_dispensed']
)

# Calculate average monthly commodities dispensed
data_df['average_monthly_commodities_dispensed'] = (
    data_df['total_commodities_dispensed'] /data_df['number_of_months']
)


# ## *c) Exploratory Data Analysis (EDA)*

# ### Descriptive Statistics

# #### i) FP service data descriptives

# In[81]:


# List of columns to describe
eda_cols = [
    'condoms_new', 'condoms_revisits',
    'pills_new', 'pills_revisits',
    'injectable_new', 'injectable_revisits',
    'implants_new', 'implants_revisits',
    'iucd_new', 'iucd_revisits',
    'surgical_new', 'surgical_revisits'
]

# Basic descriptive statistics
eda_summary = df_service1[eda_cols].describe().T
eda_summary['missing'] = df_service1[eda_cols].isnull().sum()
eda_summary['zeros'] = (df_service1[eda_cols] == 0).sum()
eda_summary['unique'] = df_service1[eda_cols].nunique()
eda_summary['dtype'] = df_service1[eda_cols].dtypes

# Show the summary
print(eda_summary)


# In[82]:


# Columns to describe
eda_columns = [
    'year_month', 'country', 'county', 'uid', 'uid_code', 'county_code',
    'adolescent_10_24_receiving_fp_new', 'adolescent_10_24_receiving_fp_revisits',
    'adults_25+_receiving_fp_services_new', 'adults_25+_receiving_fp_services_revisits',
    'condoms_new', 'condoms_revisits',
    'pills_new', 'pills_revisits',
    'injectable_new', 'injectable_revisits',
    'implants_new', 'implants_revisits',
    'iucd_new', 'iucd_revisits',
    'surgical_new', 'surgical_revisits',
    'condoms', 'pills', 'injectables', 'implants', 'iucd', 'surgical',
    'total_modern_fp', 'total_cyp'
]

# Subset the DataFrame
df_eda = df_service1[eda_columns]

# 1. General info
print("=== Basic Info ===")
print(df_eda.info())

# 2. Missing values
print("=== Missing Values ===")
print(df_eda.isnull().sum())

# 3. Descriptive statistics for numeric columns
print("=== Summary Statistics ===")
print(df_eda.describe().T)

# 4. Unique value counts for categorical columns
cat_cols = ['year_month', 'country', 'county', 'uid', 'uid_code', 'county_code']
print("=== Unique Values in Categorical Columns ===")
for col in cat_cols:
    print(f"{col}: {df_eda[col].nunique()} unique values")

# 5. Trends over time (optional)
if pd.api.types.is_datetime64_any_dtype(df_eda['year_month']):
    trend_data = df_eda.groupby('year_month')['total_fp'].sum()
    trend_data.plot(title="Total FP Over Time", figsize=(10, 4))
    plt.ylabel("Total FP")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 6. Distribution plots for select variables (optional)
num_vars = ['total_modern_fp', 'total_cyp', 'condoms', 'pills', 'injectables']
for var in num_vars:
    plt.figure(figsize=(6, 3))
    sns.histplot(df_eda[var], bins=30, kde=True)
    plt.title(f'Distribution of {var}')
    plt.tight_layout()
    plt.show()


# In[83]:


# Boxplot of total modern fp
plt.figure(figsize=(6,4))
sns.boxplot(x=data_df['total_modern_fp'].dropna(), color='salmon')
plt.title('Boxplot of total_modern_fp') # Set title
plt.xlabel('total_fp') # Set X_label
plt.tight_layout() # Spacing
plt.show()


# In[84]:


# Convert year_month values into datetime objects
data_df['year_month'] = data_df['year_month'].astype(str) # Convert to string

# Split into year and month
data_df['year'] = data_df['year_month'].str[:4].astype(int)
data_df['month'] = data_df['year_month'].str[4:].astype(int)


# ### What's the FP method mix composition for the period in focus?

# In[85]:


import matplotlib.pyplot as plt
# Group data by year_month and sum the method counts
method_mix = data_df.groupby('year_month')[['pills', 'condoms', 'injectables', 'implants', 'iucd', 'surgical']].sum()

# Calculate the total for each method across the entire period
total_method_counts = method_mix.sum()

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(total_method_counts, labels=total_method_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Majority of the FP methods provided bewteen 2013 and 2024 were short term methods\n(pills + condoms + injectables ~ 87.2%)', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# What's the FP method volume by county?

# In[86]:


# Method Mix volume by county

# Sum the total counts for each method by county
county_method_totals = data_df.groupby('county')[['pills', 'condoms', 'injectables', 'implants', 'iucd', 'surgical']].sum()

# Calculate the total for all methods per county for sorting
county_method_totals['total'] = county_method_totals.sum(axis=1)

# Sort counties by total count in descending order
county_method_totals = county_method_totals.sort_values('total', ascending=False).drop(columns='total')

# Convert values to thousands and format to 0 decimal places
county_method_totals_thousands = county_method_totals / 1000

# Plotting the stacked bar chart
plt.figure(figsize=(20, 10))
ax = county_method_totals_thousands.plot(kind='bar', stacked=True, figsize=(20, 10))

# Add labels and title
plt.xlabel('County', fontsize=14)
plt.ylabel('Total Clients (Thousands)', fontsize=14)
plt.title('Total Clients by Family Planning Method and County (2013-2024)', fontsize=16)
plt.xticks(rotation=90)
plt.legend(title='Method')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add total labels on top of bars
for i, county in enumerate(county_method_totals_thousands.index):
    total_value = county_method_totals_thousands.loc[county].sum()
    # Position the text slightly above the bar. Adjust text position as needed.
    ax.text(i, total_value + 10, f'{total_value:.0f}', ha='center', va='bottom', rotation=90, fontsize=11)


plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()


# ### What are the trends in FP Method uptake over time?

# In[87]:


# Plot FP method uptake trends over time

# Group data by year_month and sum the method counts
method_uptake_time = data_df.groupby('year_month')[['pills', 'condoms', 'injectables', 'implants', 'iucd', 'surgical']].sum()

# Convert values to thousands
method_uptake_time_thousands = method_uptake_time / 1000

# Plotting the stacked bar chart
plt.figure(figsize=(20, 10))
ax = method_uptake_time_thousands.plot(kind='bar', stacked=True, figsize=(20, 10))

# Add labels and title
plt.xlabel('Year-Month', fontsize=14)
plt.ylabel('Total Client Count (Thousands)', fontsize=14)
plt.title('Family Planning Method Uptake Trends Over Time (2013-2024) \n (Uptake increased dramatically after 2019 potentially due to reporting gaps)', fontsize=16)
plt.xticks(rotation=90, fontsize=8)
plt.legend(title='Method', loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add total labels on top of bars
for i, year_month in enumerate(method_uptake_time_thousands.index):
    total_value = method_uptake_time_thousands.loc[year_month].sum()
    # Position the text slightly above the bar. Adjust text position as needed.
    ax.text(i, total_value, f'{total_value:.0f}', ha='center', va='bottom', fontsize=6)


plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlapping
plt.show()


# ### Total FP Methods vs Women Eligible for FP Over Time

# In[88]:


# Plot Total modern FP Methods vs. Women Eligible for FP Over Time

# Group by year and sum 'total_fp', then get the maximum 'eligible_fp' for each year
trends = data_df.groupby(data_df['year'])[['total_modern_fp', 'eligible_fp']].agg(
    {'total_modern_fp': 'sum', 'eligible_fp': 'sum'}
)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(trends.index, trends['total_modern_fp'], label='Total FP Methods', marker='o')
plt.plot(trends.index, trends['eligible_fp'], label='Women (10-49 Yrs) eligible for FP' , marker='o')

# Add labels and title
plt.xlabel('Year', fontsize=12)
plt.ylabel('Users (millions)', fontsize=12)
plt.title('Total Modern FP Methods vs. Women Eligible for FP Over Time', fontsize=14)
plt.xticks(trends.index, rotation=45)
plt.legend()
plt.grid(False)

# Show plot
plt.tight_layout()
plt.show()


# ### Which FP commodities face the greatest supply-demand mismatch?

# In[89]:


# Create month column (Period) for aggregation
if 'date' not in data_df.columns:
    data_df['date'] = pd.to_datetime(data_df['year_month'], format='%Y%m')

# Define the commodity columns we need (issued / stock received)
commodities = {
    'male_condoms': ('condoms_male_condom_stock_dispensed', 'condoms_male_condom_stock_received'),
    'female_condoms': ('condoms_female_condom_stock_dispensed', 'condoms_female_condom_stock_received'),
    'pills': ('pills_combined_oral_contraceptive_stock_dispensed', 'pills_combined_oral_contraceptive_stock_received'),
    'injectables': ('injectables_stock_dispensed', 'injectables_stock_received'),
    'implants': ('implants_stock_dispensed', 'implants_stock_received')
}

# Make sure the measure columns are numeric
for issued_col, received_col in commodities.values():
    data_df[issued_col] = pd.to_numeric(data_df[issued_col], errors='coerce')
    data_df[received_col] = pd.to_numeric(data_df[received_col], errors='coerce')

# Build a tidy dataframe with national-level monthly totals for each commodity
agg_list = []
for name, (issued_col, received_col) in commodities.items():
    monthly = (
        data_df
        .groupby('date')[[issued_col, received_col]]
        .sum()
        .rename(columns={issued_col: 'issued', received_col: 'received'})
    )
    monthly['commodity'] = name
    monthly['coverage_ratio'] = monthly['issued'] / monthly['received'].replace(0, pd.NA)
    monthly['stockout_flag'] = monthly['issued'] > monthly['received']
    agg_list.append(monthly.reset_index())

commod_df = pd.concat(agg_list, ignore_index=True)

# Count the number of months where issued exceeded receipts (potential stock-out months)
stockouts = commod_df.groupby('commodity')['stockout_flag'].sum().reset_index(name='months_with_potential_stockout')
print(stockouts)

# Visualise issued vs received with stock-out shading
sns.set_style('whitegrid')
fig, axes = plt.subplots(len(commodities), 1, figsize=(14, 4 * len(commodities)), sharex=True)
for ax, (name, grp) in zip(axes, commod_df.groupby('commodity')):
    ax.plot(grp['date'], grp['issued'], label='Issued', color='steelblue')
    ax.plot(grp['date'], grp['received'], label='Received', color='orange')
    ax.fill_between(grp['date'], grp['issued'], grp['received'], where=grp['stockout_flag'], color='red', alpha=0.25, label='Potential stock-out')
    ax.set_title(name.replace('_', ' ').title())
    ax.legend()
plt.tight_layout()
plt.show()


# ### How well do clients stay on or return for each family planning method once they start, and which methods need the most support to improve continuation?

# In[90]:


# New vs revisits columns
methods = {
    'condoms': ('condoms_new', 'condoms_revisits'),
    'pills': ('pills_new', 'pills_revisits'),
    'injectables': ('injectable_new', 'injectable_revisits'),
    'implants': ('implants_new', 'implants_revisits'),
    'iucd': ('iucd_new', 'iucd_revisits')
}

data_df['date'] = pd.to_datetime(data_df['year_month'], format='%Y%m')

for pair in methods.values():
    for col in pair:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

cont_list = []
for name, (new_col, rev_col) in methods.items():
    monthly = data_df.groupby('date')[[new_col, rev_col]].sum().rename(columns={new_col: 'new', rev_col: 'revisit'})
    monthly['method'] = name
    monthly['continuation_rate'] = monthly['revisit'] / (monthly['new'] + monthly['revisit']).replace(0, pd.NA)
    cont_list.append(monthly.reset_index())

cont_df = pd.concat(cont_list)

# Summary continuation stats
cont_summary = cont_df.groupby('method')['continuation_rate'].describe()[['mean','50%','min','max']]
print(cont_summary)

# Plot continuation over time for injectables and implants (common methods)
plt.figure(figsize=(14,6))
for method in ['injectables','implants','condoms','pills', 'iucd']:
    subset = cont_df[cont_df['method'] == method]
    plt.plot(subset['date'], subset['continuation_rate'], label=method)
plt.title('Monthly continuation rate (revisit share)')
plt.ylabel('Continuation rate')
plt.xlabel('Date')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# In[91]:




correlation = data_df['total_users_receiving_fp'] .corr(data_df['total_fp'])
print(f"Correlation between total_users_receiving_fp and total_fp: {correlation:.2f}")


# ### Proportion of adolescents receiving FP and Adults receiving FP

# In[92]:


# Sum totals
adolescents_total = df_service1['adolescent_10_24_receiving_fp_new'].sum()
adults_total = df_service1['adults_25+_receiving_fp_services_new'].sum()
total = adolescents_total + adults_total

# Calculate proportions
adolescents_prop = adolescents_total / total
adults_prop = adults_total / total

# Display
print(f"Proportion of Adolescents (10–24) receiving FP: {adolescents_prop:.2%}")
print(f"Proportion of Adults (25+) receiving FP: {adults_prop:.2%}")


# In[93]:


# Data
labels = ['Adolescents (10–24)', 'Adults (25+)']
sizes = [adolescents_total, adults_total]
colors = ['#66b3ff', '#ff9999']

# Plot
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Proportion of Adolescents vs Adults Receiving FP')
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular.
plt.show()


# ### Modern Contraceptive Prevalence Rate(mCPR) vs Unmet need

# In[94]:


fig = plt.figure(figsize=(20, 16))
fig.suptitle('Analysis of Unmet Need vs Modern Contraceptive Prevalence Rate (mCPR)', fontsize=20)

mcpr_col = 'mCPR (Married Women, %)'
unmet_need_col = 'Total Unmet Need (Married Women, %)'
demand_satisfied_col = 'Demand_Satisfied_by_Modern_Methods (%)'

# Filter to rows that have both mCPR and unmet need data
complete_data = data_df.dropna(subset=[mcpr_col, unmet_need_col])
print(f"Number of rows with both mCPR and unmet need data: {len(complete_data)}")

# Calculate service statistics-based mCPR
if 'total_modern_fp' in data_df.columns and 'eligible_fp' in data_df.columns:
    data_df['calculated_mcpr'] = (data_df['total_modern_fp'] / data_df['eligible_fp']) * 100

# 2. Create county-level summary for comparison
county_summary = data_df.groupby('county').agg({
    'total_modern_fp': 'sum',
    'eligible_fp': 'sum',
    'traditional': 'sum',
    mcpr_col: 'mean',
    unmet_need_col: 'mean',
    demand_satisfied_col: 'mean'
}).reset_index()

# Calculate mCPR at county level based on service statistics
county_summary['calculated_mcpr'] = (county_summary['total_modern_fp'] / county_summary['eligible_fp']) * 100

# Filter to counties with complete data
counties_with_data = county_summary.dropna(subset=[mcpr_col, unmet_need_col])
print(f"Number of counties with both mCPR and unmet need data: {len(counties_with_data)}")

# Initialize correlation variable
correlation = 0

# If we have counties with both metrics available
if len(counties_with_data) > 0:
    # 3. Scatterplot of mCPR vs Unmet Need
    ax1 = fig.add_subplot(2, 2, 1)
    scatter = sns.scatterplot(
        x=mcpr_col, 
        y=unmet_need_col, 
        data=counties_with_data,
        s=100, 
        ax=ax1
    )
    
    # Add county labels to points
    for i, row in counties_with_data.iterrows():
        ax1.annotate(row['county'], 
                    (row[mcpr_col], row[unmet_need_col]),
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    # Calculate correlation
    correlation = counties_with_data[[mcpr_col, unmet_need_col]].corr().iloc[0, 1]
    ax1.set_title(f'Unmet Need vs mCPR by County\nCorrelation: {correlation:.2f}', fontsize=14)
    ax1.set_xlabel('mCPR (Married Women, %)', fontsize=12)
    ax1.set_ylabel('Total Unmet Need (Married Women, %)', fontsize=12)
    
    # Add regression line
    sns.regplot(x=mcpr_col, y=unmet_need_col, data=counties_with_data, 
                scatter=False, ax=ax1, color='red')

    # 4. Relationship with Demand Satisfied
    if demand_satisfied_col in counties_with_data.columns:
        ax2 = fig.add_subplot(2, 2, 2)
        demand_corr = counties_with_data[[mcpr_col, demand_satisfied_col]].corr().iloc[0, 1]
        
        sns.scatterplot(
            x=mcpr_col, 
            y=demand_satisfied_col, 
            data=counties_with_data,
            s=100, 
            ax=ax2
        )
        
        # Add regression line
        sns.regplot(x=mcpr_col, y=demand_satisfied_col, data=counties_with_data, 
                    scatter=False, ax=ax2, color='red')
        
        ax2.set_title(f'Demand Satisfied vs mCPR\nCorrelation: {demand_corr:.2f}', fontsize=14)
        ax2.set_xlabel('mCPR (Married Women, %)', fontsize=12)
        ax2.set_ylabel('Demand Satisfied by Modern Methods (%)', fontsize=12)

# 5. Compare calculated mCPR with unmet need
counties_with_calculated = county_summary.dropna(subset=[unmet_need_col]).copy()
if len(counties_with_calculated) > 0:
    ax3 = fig.add_subplot(2, 2, 3)
    calculated_corr = np.nan
    
    # Only calculate correlation if we have enough data points
    if len(counties_with_calculated) >= 3:
        calculated_corr = counties_with_calculated[['calculated_mcpr', unmet_need_col]].corr().iloc[0, 1]
        
    sns.scatterplot(
        x='calculated_mcpr', 
        y=unmet_need_col, 
        data=counties_with_calculated,
        s=100, 
        ax=ax3
    )
    
    for i, row in counties_with_calculated.iterrows():
        ax3.annotate(row['county'], 
                    (row['calculated_mcpr'], row[unmet_need_col]),
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    # Add regression line if we have enough data points
    if len(counties_with_calculated) >= 3:
        sns.regplot(x='calculated_mcpr', y=unmet_need_col, data=counties_with_calculated, 
                    scatter=False, ax=ax3, color='red')
    
    title = f'Unmet Need vs Calculated mCPR\nCorrelation: '
    if not np.isnan(calculated_corr):
        title += f"{calculated_corr:.2f}"
    else:
        title += "N/A"
    ax3.set_title(title, fontsize=14)
    ax3.set_xlabel('Calculated mCPR (based on service statistics, %)', fontsize=12)
    ax3.set_ylabel('Total Unmet Need (Married Women, %)', fontsize=12)

# 6. Estimate unmet need for counties that have demand satisfied data
ax4 = fig.add_subplot(2, 2, 4)

counties_with_demand = county_summary.dropna(subset=[demand_satisfied_col]).copy()
if len(counties_with_demand) > 0:
    # Estimate unmet need using the formula:
    # Unmet Need = (mCPR / Demand Satisfied) - mCPR
    # Or alternatively: mCPR * ((100 / Demand Satisfied) - 1)
    counties_with_demand['estimated_unmet_need'] = counties_with_demand['calculated_mcpr'] *         ((100 / counties_with_demand[demand_satisfied_col]) - 1)
    
    # Sort by estimated unmet need
    top_unmet_need = counties_with_demand.sort_values('estimated_unmet_need', ascending=False).head(15)
    
    # Create bar chart
    sns.barplot(
        x='estimated_unmet_need', 
        y='county', 
        data=top_unmet_need,
        ax=ax4
    )
    
    # Add calculated mCPR as text annotations
    for i, row in enumerate(top_unmet_need.itertuples()):
        ax4.text(
            row.estimated_unmet_need + 0.2, 
            i, 
            f'mCPR: {row.calculated_mcpr:.1f}%', 
            va='center'
        )
    
    ax4.set_title('Top 15 Counties by Estimated Unmet Need', fontsize=14)
    ax4.set_xlabel('Estimated Unmet Need (%)', fontsize=12)
    ax4.set_ylabel('County', fontsize=12)

# Add a textbox with summary findings
text_x = 0.5
text_y = 0.02
summary_text = ("Summary Findings:\n"
               "1. There is a negative correlation between mCPR and unmet need, as expected.\n"
               "2. Counties with lower mCPR tend to have higher unmet need for family planning.\n"
               "3. Service statistics show considerable variation in calculated mCPR across counties.\n"
               "4. Estimated unmet need is highest in counties with very low mCPR values.")

fig.text(text_x, text_y, summary_text, ha='center', va='bottom', 
         fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])


# Output detailed data
print("\n=== DETAILED RESULTS ===")
print("\nTop 10 counties by unmet need:")
if len(counties_with_data) > 0:
    print(counties_with_data.sort_values(unmet_need_col, ascending=False)[['county', mcpr_col, unmet_need_col, demand_satisfied_col]].head(10).to_string(index=False))
else:
    print("No counties have both mCPR and unmet need data")

print("\nTop 10 counties by estimated unmet need:")
if len(counties_with_demand) > 0:
    print(counties_with_demand.sort_values('estimated_unmet_need', ascending=False)[['county', 'calculated_mcpr', 'estimated_unmet_need']].head(10).to_string(index=False))
else:
    print("No counties have data to estimate unmet need")

print("\nCounties with lowest mCPR:")
if len(counties_with_data) > 0:
    print(counties_with_data.sort_values(mcpr_col)[['county', mcpr_col, unmet_need_col]].head(10).to_string(index=False))
else:
    print(county_summary.sort_values('calculated_mcpr')[['county', 'calculated_mcpr']].head(10).to_string(index=False))

print("\nAnalysis complete. See '/workspace/unmet_need_vs_mcpr_analysis.png' for visualization.")


# ## *d) Modelling*

# ### Baseline Model_Linear Regression

# In[187]:


# Target and feature columns
target_cols = [
    'total_actual_revisits_modern_fp_methods',
    'total_actual_modern_fp',
    'total_fp',
    'proportion_adolescents_10_24_yrs_receiving_fp',
    'actual_core_health_workers_2013_10000_eligible_fp',
    'average_monthly_commodities_dispensed',
]

feature_cols= feature_cols = ['year','year_month','county','number_of_months']


# In[188]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[190]:



#  Identify numeric features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Define preprocessing steps for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),       # Handle missing values
    ('scaler', StandardScaler())                         # Feature scaling
])

# Categorical preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers 
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

#  Define full pipeline with estimator
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(LinearRegression()))
])

#  Fit the pipeline 
model_pipeline.fit(X_train, y_train.fillna(0))  # Fill y_train NaNs with 0 or another value if needed

#  Predict on test set 
y_pred = model_pipeline.predict(X_test)


# In[191]:


# Evaluate each target separately
for i, column in enumerate(y.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{column}:")
    print(f"  MSE = {mse:.2f}")
    print(f"  R²  = {r2:.2f}")
    print("-" * 40)


# ### XGBoost

# In[199]:


# Preprocessing for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline with XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )))
])

# Hyperparameter grid
param_grid = {
    'regressor__estimator__n_estimators': [50, 100],
    'regressor__estimator__learning_rate': [0.05, 0.1],
    'regressor__estimator__max_depth': [3, 5],
    'regressor__estimator__subsample': [0.8, 1.0],
    'regressor__estimator__colsample_bytree': [0.8, 1.0]
}

# Grid search with 3-fold cross-validation
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)

# Fit grid search
grid_search.fit(X_train, y_train)

# Fit model
xgb_pipeline.fit(X_train, y_train)

# Predict
y_pred = xgb_pipeline.predict(X_test)


# In[200]:


# Evaluate the model
for i, col in enumerate(target_cols):
    mse = mean_squared_error(y_test[col], y_pred[:, i])
    r2 = r2_score(y_test[col], y_pred[:, i])
    print(f"{col}:\n  MSE = {mse:.2f}\n  R²  = {r2:.2f}\n" + "-"*40)


# **Observation**
# * XGBoost improved overall performance for some targets but underperformed for others.
# * The workforce target remains the most predictable — good sign.
# * Commodities dispensed remains poorly predicted

# In[203]:


import joblib

# Save model
joblib.dump(XGBRegressor, 'my_model.pkl')

# Load model
model = joblib.load('my_model.pkl')


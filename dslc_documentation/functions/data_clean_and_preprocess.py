import pandas as pd
import numpy as np
import random as rd

def data_clean_and_preprocess_eda(diabetes):
    #Clean the numberic variables

    #Reading the reference, "age" >= 85 means the individual's age is uncertain, then drop.
    diabetes = diabetes[diabetes["age"] <= 84]
    #Reading the reference, "bmi" >= 9995 means the individual's "bmi" is uncertain, then drop.
    diabetes = diabetes[diabetes["bmi"] <= 9994]
    #Reading the reference, "weight" > 299 means the individual's weight can't be record or uncertain, then drop.
    diabetes = diabetes[(diabetes["weight"] <= 299) & (diabetes["weight"] >= 100)]
    #Reading the reference, "height" > 76 or < 59 means the individual's height can't be record or unknown, then drop.
    diabetes = diabetes[(diabetes["height"] <= 76) & (diabetes["height"] >= 59)]

    #Clean the category variables
    #Reading the reference, "smoker" != 1 or 2 means the individual didn't provide certain information, then drop.
    diabetes = diabetes[diabetes["smoker"] <= 2]
    #Other binary variables can only have values of 0 or 1, so don't preprocess them.

    #Delete the rows with missing values
    diabetes.dropna()

    #Adjust the values of some binary variables to make the values are only 0 or 1.
    diabetes["smoker"] = diabetes["smoker"] - 1
    diabetes["sex"] = diabetes["sex"] - 1

    return diabetes

def data_clean_and_preprocess_default(diabetes):
    #Clean the numberic variables

    if "bmi" in diabetes.columns:
        diabetes = diabetes.drop(columns = ["bmi"])

    #Reading the reference, "age" >= 85 means the individual's age is uncertain, then drop.
    diabetes = diabetes[diabetes["age"] <= 84]
    #Reading the reference, "bmi" >= 9995 means the individual's "bmi" is uncertain, then drop.
    #diabetes = diabetes[diabetes["bmi"] <= 9994]
    #Reading the reference, "weight" > 299 means the individual's weight can't be record or uncertain, then drop.
    diabetes = diabetes[(diabetes["weight"] <= 299) & (diabetes["weight"] >= 100)]
    #Reading the reference, "height" > 76 or < 59 means the individual's height can't be record or unknown, then drop.
    diabetes = diabetes[(diabetes["height"] <= 76) & (diabetes["height"] >= 59)]

    #Clean the category variables
    
    #Reading the reference, "smoker" != 1 or 2 means the individual didn't provide certain information, then drop.
    diabetes = diabetes[diabetes["smoker"] <= 2]
    #Other binary variables can only have values of 0 or 1, so don't preprocess them.

    #The following are in the first version of function
    #Reading the reference, "hypertension" != 1 or 2 means the individual didn't provide certain information, then drop.
    #diabetes = diabetes[diabetes["hypertension"] <= 2]
    #Reading the reference, "heart_condition" != 1 or 2 means the individual didn't provide certain information, then drop.
    #diabetes = diabetes[diabetes["heart_condition"] <= 2]
    #Reading the reference, "cancer" != 1 or 2 means the individual didn't provide certain information, then drop.
    #diabetes = diabetes[diabetes["cancer"] <= 2]
    #Reading the reference, "family_history_diabetes" != 1 or 2 means the individual didn't provide certain information, then drop.
    #diabetes = diabetes[diabetes["family_history_diabetes"] <= 2]

    diabetes.dropna()

    diabetes["smoker"] = diabetes["smoker"] - 1
    diabetes["sex"] = diabetes["sex"] - 1

    return diabetes
    


def data_clean_and_preprocess_perturb(diabetes_o, limit_process = False, smoker_process = False, unknown_process = False, bmi_keep = False, standard = False):

    diabetes = diabetes_o.copy()

    if bmi_keep == False and "bmi" in diabetes.columns:
        diabetes = diabetes.drop(columns = ["bmi"])

        
    if limit_process:
        diabetes["high age"] = (diabetes["age"] == 85)
        diabetes["high weight"] = (diabetes["weight"] == 996)
        diabetes["high height"] = (diabetes["height"] == 96)
        if bmi_keep and "bmi" in diabetes.columns:
            diabetes["high bmi"] = (diabetes["bmi"] == 9995)
        for i in diabetes.index:
            if diabetes.loc[i]["height"] == 96:
                diabetes.at[i,"height"] = 77
            if diabetes.loc[i]["weight"] == 996:
                diabetes.at[i,"weight"] = 300
    else:
        diabetes = diabetes[diabetes["age"] != 85]
        if bmi_keep and "bmi" in diabetes.columns:
            diabetes = diabetes[diabetes["bmi"] != 9995]
        diabetes = diabetes[diabetes["weight"] != 996]
        diabetes = diabetes[diabetes["height"] != 96]

    if unknown_process:
        dbt_copy = diabetes.copy()
        dbt_copy = dbt_copy[dbt_copy["age"] <= 84]
        if bmi_keep and "bmi" in diabetes.columns:
            dbt_copy = dbt_copy[dbt_copy["bmi"] <= 9994]
        dbt_copy = dbt_copy[(dbt_copy["weight"] <= 299) & (dbt_copy["weight"] >= 100)]
        dbt_copy = dbt_copy[(dbt_copy["height"] <= 76) & (dbt_copy["height"] >= 59)]
        means = dbt_copy.drop(columns = ["house_family_person_id"]).mean()
        std = dbt_copy.drop(columns = ["house_family_person_id"]).std()
        for i in diabetes.index:
            if diabetes.loc[i]["height"] > 96: 
                diabetes.at[i,"height"] = int(np.random.normal(means["height"], std["height"]))
            if bmi_keep and "bmi" in diabetes.columns:
                if diabetes.loc[i]["bmi"] == 9999: 
                    diabetes.at[i,"bmi"] = int(np.random.normal(means["bmi"], std["bmi"]))
            if diabetes.loc[i]["weight"] > 996: 
                diabetes.at[i,"weight"] = int(np.random.normal(means["weight"], std["weight"]))
    else:
        if bmi_keep and "bmi" in diabetes.columns:
            diabetes = diabetes[diabetes["bmi"] <= 9995]
        diabetes = diabetes[(diabetes["weight"] <= 300) & (diabetes["weight"] >= 100)]
        diabetes = diabetes[(diabetes["height"] <= 77) & (diabetes["height"] >= 59)]

    if standard:
        means = diabetes.drop(columns = ["house_family_person_id"]).mean()
        std = diabetes.drop(columns = ["house_family_person_id"]).std()
        diabetes["height"] = (diabetes["height"] - means["height"])/std["height"]
        diabetes["weight"] = (diabetes["weight"] - means["weight"])/std["weight"]
        if bmi_keep and "bmi" in diabetes.columns:
            diabetes["bmi"] = (diabetes["bmi"] - means["bmi"])/std["bmi"]
        diabetes["age"] = (diabetes["age"] - means["age"])/std["age"]

    if smoker_process:
        diabetes["smoke"] = diabetes["smoker"] == 1
        diabetes["no smoke"] = diabetes["smoker"] == 2
        diabetes["unsure smoke"] = diabetes["smoker"] >= 7
        diabetes = diabetes.drop(columns = ["smoker"])
    else:
        diabetes = diabetes[diabetes["smoker"] <= 2]
        diabetes["smoker"] = diabetes["smoker"] - 1


    diabetes.dropna()
    diabetes["sex"] = diabetes["sex"] - 1

    return diabetes

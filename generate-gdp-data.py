import math
import pickle
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

DATA_DIR = Path("datasets/original_csv")

def seralize(name, data, training):
    if name == 'adult':
        return adult_sf(data, training)
    elif name == 'bank':
        return bank_sf(data, training)
    elif name == 'blood':
        return blood_sf(data, training)
    elif name == 'calhousing':
        return calhousing_sf(data, training)
    elif name == 'car':
        return car_sf(data, training)
    elif name == 'diabetes':
        return diabetes_sf(data, training)
    elif name == 'heart':
        return heart_sf(data, training)
    elif name == 'jungle':
        return jungle_sf(data, training)
    else:
        print("Please select correct dataset name: adult, bank, blood, calhousing, car, diabetes, heart, jungle")
        exit()

def adult_sf(data, training=True):
    prompts = []
    testing_labels= []
        
    for index, row in data.iterrows():
        age, workclass, education, marital, occupation, relationship, race, sex, gain, loss, hours, country, label = row
        
        # Pronouns
        if sex=='Female':
            pro = ['She', 'Her', 'Her', 'Hers'] 
        else:
            pro = ['He', 'Him', 'His', 'His']
            
        # Value-based prompts for categorical attributes
        pr_workclass = {
            'State-gov': f'{pro[0]} works in state government.',
            'Self-emp-not-inc': f'{pro[0]} is self-employed in an unincorporated business.',
            'Private': f'{pro[0]} works in the private sector.',
            'Federal-gov': f'{pro[0]} works in federal government.',
            'Local-gov': f'{pro[0]} works in local government.',
            'Self-emp-inc': f'{pro[0]} is self-employed in an incorporated business.',
            'Without-pay': f'{pro[0]} works without pay.',
            'Never-worked': f'{pro[0]} has never worked.'
        }
        
        pr_education = {
            'Bachelors': f'{pro[0]} has a Bachelor\'s degree at most.',
            'HS-grad': f'{pro[0]} has a high school degree at most.', 
            '11th': f'{pro[0]} has completed eleventh grade at most.', 
            'Masters': f'{pro[0]} has a Master\'s degree at most.', 
            '9th': f'{pro[0]} has completed ninth grade at most.', 
            'Some-college': f'{pro[0]} has attended some college at most.', 
            'Assoc-acdm': f'{pro[0]} has an academic associate\'s degree at most.', 
            'Assoc-voc': f'{pro[0]} has a vocational associate\'s degree at most.',  
            '7th-8th': f'{pro[0]} has completed seventh or eighth grade at most.',  
            'Doctorate': f'{pro[0]} has a doctorate degree at most.', 
            'Prof-school': f'{pro[0]} has attended a professional school at most.', 
            '5th-6th': f'{pro[0]} has completed fifth or sixth grade at most.',  
            '10th': f'{pro[0]} has completed tenth grade at most.', 
            '1st-4th': f'{pro[0]} has completed fourth grade at most.',  
            'Preschool': f'{pro[0]} has attended at most pre-school at most.', 
            '12th': f'{pro[0]} has completed twelfth grade at most.', 
        }
        
        pr_marital = {
            'Never-married': f'{pro[0]} has never been married.', 
            'Married-civ-spouse': f'{pro[0]} is married to a civilian.',
            'Divorced': f'{pro[0]} is divorced.',
            'Married-spouse-absent': f'{pro[0]} is married to an absent spouse.',
            'Separated': f'{pro[0]} is seperated from their spouse.', # legally seperated
            'Married-AF-spouse': f'{pro[0]} is married to a spouse in the Armed Forces.', 
            'Widowed': f'{pro[0]} is widowed.',
        }
        
        pr_occupation = {
            'Adm-clerical': f'{pro[2]} occupation is in clerical administration.',
            'Exec-managerial': f'{pro[2]} occupation is in executive management.',
            'Handlers-cleaners': f'{pro[2]} occupation is in handling and/or cleaning.',
            'Prof-specialty': f'{pro[2]} occupation is in professional specialty.',
            'Other-service': f'{pro[2]} occupation is in the service industry.',
            'Sales': f'{pro[2]} occupation is in sales.',
            'Craft-repair': f'{pro[2]} occupation is in craft and/or repair.',
            'Transport-moving': f'{pro[2]} occupation is in transporation and/or moving.',
            'Farming-fishing': f'{pro[2]} occupation is in farming and/or fishing.',
            'Machine-op-inspct': f'{pro[2]} occupation is in machine operation and inspection.',
            'Tech-support': f'{pro[2]} occupation is in technical support.',
            'Protective-serv': f'{pro[2]} occupation is in protective services.',
            'Armed-Forces': f'{pro[2]} occupation is in the Armed Forces.',
            'Priv-house-serv': f'{pro[2]} occupation is in private house service.'
        }
        
        pr_relationship = {
            'Not-in-family': f'{pro[0]} is not related to the other person in {pro[2].lower()} household.',
            'Husband': f'{pro[0]} is the husband of the other person in {pro[2].lower()} household.',
            'Wife': f'{pro[0]} is the wife of the other person in {pro[2].lower()} household.',
            'Own-child': f'{pro[0]} is the child of the other person in {pro[2].lower()} household.',
            'Unmarried': f'{pro[0]} is not married to the other person in {pro[2].lower()} household.',
            'Other-relative': f'{pro[0]} is a relative of the other person in {pro[2].lower()} household.'
        }
            
        pr_race = {
            'White': f'{pro[0]} is White.',
            'Black': f'{pro[0]} is Black.',
            'Asian-Pac-Islander': f'{pro[0]} is Asian or Pacific Islander.',
            'Amer-Indian-Eskimo': f'{pro[0]} is American Indian or Eskimo.',
            'Other': f'{pro[0]} is not White or Black or Asian or Pacific Islander or American Indian or Eskimo.',
        }
        
        # General prompts for numeric attributes
        try:
            pr_age_sex = f'This person is a {np.clip(round(age), a_min=0, a_max=120)} years old {sex.lower()}.'
        except AttributeError:
            pr_age_sex = f'This person is a {np.clip(round(age), a_min=0, a_max=120)} years old {sex}.'
        pr_capital = f'{pro[0]} had a capital gain of {np.round(gain,2)} and a capital loss of {np.round(loss,2)} last year.'
        pr_hours = f'{pro[0]} works {int(np.clip(hours, a_min=0, a_max=120))} hours per week.'
        
        if country=='Outlying-US(Guam-USVI-etc)':
            pr_native = f'{pro[0]} is from U.S. Territories and Minor Outlying Islands.'  
        else :
            pr_native = f'{pro[0]} is from {country}.'
            
        if label == True:
            pr_label = 'Does this person earn more than 50,000 dollars annually? Yes or No? Answer: Yes'
            # pr_label = f'{pro[0]} earns more than 50,000 dollars annually.'
        else:
            pr_label = 'Does this person earn more than 50,000 dollars annually? Yes or No? Answer: No'
            # pr_label = f'{pro[0]} earns less than 50,000 dollars annually.'

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        # Get prompt strings
        prompt = 'An individual recorded in the 1994 US census is described as follows:' +\
        f' {pr_age_sex} {pr_education[education]} {pr_workclass[workclass]}' +\
        f' {pr_occupation[occupation]} {pr_hours} {pr_capital} {pr_native} {pr_race[race]}' +\
        f' {pr_marital[marital]} {pr_relationship[relationship]}'

        if training:
            training_prompt = prompt + ' ' + pr_label
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Does this person earn more than 50,000 dollars annually? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)
            
    return prompts, testing_labels

def bank_sf(data, training=True):
    prompts = []
    testing_labels= []
    
    for index, row in data.iterrows():
        age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, label = row

        pr_job = {
            'blue-collar': 'They have a blue-collar job.',
            'management': 'Their job is in management.',
            'services': 'Their job is in services.',
            'unemployed': 'They are unemployed.',
            'retired': 'They are retired.',
            'admin.': 'Their job is in administration.',
            'housemaid': 'Their job is a house-maid.',
            'technician': 'Their job is a technician.',
            'entrepreneur': 'They are an entrepeneur.',
            'student': 'They are a student.',
            'self-employed': 'They are self-employed.',
            'unknown': 'Their job is not known.'
        }
        
        pr_education = {
            'secondary': 'They have attained secondary-level education.',
            'tertiary': 'They have attained tertiary-level education.',
            'primary': 'They have attained primary-level education.',
            'unknown': 'Their education status is unknown.'
        }
        
        pr_default = {
            'no': 'They do not have credit in default.',
            'yes': 'They have credit in default.'
        }
        
        pr_housing = {
            'no': 'They do not have any housing loans.',
            'yes': 'They have housing loans.'
        }
        
        pr_loan = {
            'no': 'They do not have any personal loans.',
            'yes': 'They have personal loans.'
        }
        
        pr_outcome = {
            'unknown': 'The outcome of the previous marketing campaign is not known for this client.',
            'failure': 'The outcome of the previous marketing campaign was failure for this client.',
            'success': 'The outcome of the previous marketing campaign was success for this client.',
            'other': 'The outcome of the previous marketing campaign not success nor failure nor unknown for this client.'
        }
        
        pr_label = {
            False: 'Does this person subscribe to a term deposit? Yes or No? Answer: No',
            True: 'Does this person subscribe to a term deposit? Yes or No? Answer: Yes'
        }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        months = {'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April', 'may': 'May', 
                  'jun': 'June', 'jul': 'July', 'aug': 'August', 'sep': 'September', 'oct': 'October', 
                  'nov': 'November', 'dec': 'December'}
        
        pr_age = f'The client is {np.clip(round(age), a_min=0, a_max=120)} years old.'
        pr_marital = f'They are {marital}.'
        pr_balance = f'Their average yearly balance is {np.round(balance,2)} euros.'
        pr_contact = f'The contact communication type is {contact}.'
        pr_last = f'They were last contacted on {months[month]} {np.clip(round(day), a_min=1, a_max=31)} for a duration of {np.clip(round(duration), a_min=0, a_max=None)} seconds.'
        pr_campaign = f'They were contacted {np.clip(round(campaign), a_min=0, a_max=None)} times during this campaign.'
        
        if pdays < 0:
            pr_previous = 'They were not contacted in previous campaigns.'
        else:
            pr_previous = f'They were contacted {np.clip(round(previous), a_min=0, a_max=None)} times in a previous campaign and were last contacted {np.clip(round(pdays), a_min=0, a_max=None)} days ago for the previous campaign.'
            
        prompt = 'A client at a Portuguese banking institution is described as follows:' +\
        f' {pr_age} {pr_education[education]} {pr_job[job]} {pr_marital} {pr_balance}' +\
        f' {pr_default[default]} {pr_housing[housing]} {pr_loan[loan]} {pr_contact} {pr_campaign}' +\
        f' {pr_last} {pr_previous} {pr_outcome[poutcome]}'

        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Does this person subscribe to a term deposit? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)
            
    return prompts, testing_labels

def blood_sf(data, training=True):
    prompts = []
    testing_labels= []
    for index, row in data.iterrows():
        recency, frequency, monetray, time, label = row

        pr_frequency = f'The donor has donated blood {np.clip(round(frequency), a_min=0, a_max=None)} times.'
        pr_monetray = f'They have donated a total of {np.clip(round(monetray,2), a_min=0, a_max=None)} c.c. of blood.'
        pr_recency = f'They last donated blood {np.clip(round(recency), a_min=0, a_max=None)} months ago.'
        pr_time = f'Their first blood donation was {np.clip(round(time), a_min=0, a_max=None)} months ago.'
        # pr_label = {
        #     False: 'The donor did not donate blood in March 2007.',
        #     True: 'The donor donated blood in March 2007.'
        # }
        pr_label = {
            False: 'Did the donor donate blood in March 2007? Yes or No? Answer: No',
            True: 'Did the donor donate blood in March 2007? Yes or No? Answer: Yes'
        }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        prompt = 'A blood donor at a Blood Transfusion Service Center is described as follows:' +\
        f' {pr_frequency} {pr_monetray} {pr_recency} {pr_time}'

        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Did the donor donate blood in March 2007? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def calhousing_sf(data, training=True):
    prompts = []
    testing_labels= []
    for index, row in data.iterrows():
        label, income, age, rooms, bedrooms, population, households, latitude, longitude = row

        pr_loc = f'This housing block is located at latitude {np.clip(np.round(latitude,2), a_min=32, a_max=42)} and longitude {np.clip(np.round(longitude,2), a_min=-125, a_max=-115)}.'
        pr_rooms = f'The houses in the block have a total of {np.clip(round(rooms), a_min=np.clip(round(bedrooms), a_min=0, a_max=None), a_max=None)} rooms with {np.clip(round(bedrooms), a_min=0, a_max=np.clip(round(rooms), a_min=0, a_max=None))} bedrooms.'
        pr_median = f'The median age of houses in the block is {np.clip(round(age), a_min=0, a_max=None)} years.'
        pr_households = f'There are a total of {np.clip(round(households), a_min=0, a_max=None)} households in the block with a total population of {np.clip(round(population), a_min=0, a_max=None)}.'
        pr_income = f'The median income of the households in the block is {np.clip(round(income*10), a_min=0, a_max=None)} thousand dollars.'
        
        # pr_label = {False: 'This housing block is not valuable.',
        #             True: 'This housing block is valuable.'
        #             }
        
        pr_label = {
            False: 'Is this housing block valuable? Yes or No? Answer: No',
            True: 'Is this housing block valuable? Yes or No? Answer: Yes'
        }
        
        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        prompt = 'A house block in California has the following attributes according to the 1990 California census.' +\
        f' {pr_loc} {pr_rooms} {pr_median} {pr_households} {pr_income}'

        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Is this housing block valuable? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def car_sf(data, training=True):
    prompts = []
    testing_labels = []
    
    for index, row in data.iterrows():
        buying, maint, doors, persons, lug_boot, safety, label = row
    
        pr_buying = {
            'low': f'The buying price of this car is low.',
            'vhigh': f'The buying price of this car is very high.',
            'med': f'The buying price of this car is medium.',
            'high': f'The buying price of this car is high.',
        }
        
        pr_maint = {
            'low': f'The maintenance cost for this car is low.',
            'vhigh': f'The maintenance cost for this car is very high.',
            'med': f'The maintenance cost for this car is medium.',
            'high': f'The maintenance cost for this car is high.',
        }
        
        pr_doors = {
            '5more': 'The car has 5 or more doors.',
            '2': 'The car has 2 doors.',
            '3': 'The car has 3 doors.',
            '4': 'The car has 4 doors.'
        }
        
        pr_persons = {
            'more': 'The car can fit more than 4 people.',
            '2': 'The car can fit 2 people.',
            '4': 'The car can fit 4 people.'
        }
        
        pr_lug = {
            'big': 'The luggage boot in this car is big.',
            'small': 'The luggage boot in this car is small.',
            'med': 'The luggage boot in this car is medium-sized.'
        }
        
        pr_safety = {
            'high': 'The safety rating of this car is estimated to be high.',
            'med': 'The safety rating of this car is estimated to be medium.',
            'low': 'The safety rating of this car is estimated to be low.'
        }
        
        # pr_label = {
        #     0: 'There are four possible ratings for cars: unacceptable, acceptable, good, and very good. This car is rated to be unacceptable.',
        #     1: 'There are four possible ratings for cars: unacceptable, acceptable, good, and very good. This car is rated to be acceptable.',
        #     2: 'There are four possible ratings for cars: unacceptable, acceptable, good, and very good. This car is rated to be good.',
        #     3: 'There are four possible ratings for cars: unacceptable, acceptable, good, and very good. This car is rated to be very good.'
        # }

        pr_label = {
            0: 'Is this car acceptable? Yes or No? Answer: No',
            1: 'Is this car acceptable? Yes or No? Answer: Yes',
            2: 'Is this car acceptable? Yes or No? Answer: Yes',
            3: 'Is this car acceptable? Yes or No? Answer: Yes',
        }

        if training == False:
            pr_label = {
                0: 'No',
                1: 'Yes',
                2: 'Yes',
                3: 'Yes',
            }
        
        prompt = 'A car is described as follows:' +\
        f' {pr_buying[buying]} {pr_maint[maint]} {pr_doors[doors]} {pr_persons[persons]}' +\
        f' {pr_lug[lug_boot]} {pr_safety[safety]}'
       
        if training:
            training_prompt = prompt + ' ' + pr_label[np.clip(round(label), a_min=0, a_max=3)]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Is this car acceptable? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label[label])

    return prompts, testing_labels

def diabetes_sf(data, training=True):
    prompts = []
    testing_labels = []
    for index, row in data.iterrows():
        pregnancy, glucose, bp, skin, insulin, bmi, diabetes, age, label = row
        
        # Context: In particular, all patients here are females at least 21 years old of Pima Indian heritage.

        pr_age = f'This patient is {np.clip(round(age), a_min = 0, a_max=100)} years old.'
        pr_pregnancy = f'She has been pregnant {np.clip(round(pregnancy), 0, a_max=None)} times.'
        pr_glucose = f'Her plasma glucose concentration at two hours in an oral glucose tolerance test' +\
        f' is {np.clip(np.round(glucose,2), a_min=0, a_max=None)} milligrams per deciliter.'
        pr_bp = f'Her blood pressure is measured to be {np.clip(np.round(bp,2), a_min=0, a_max=None)} mm Hg.'
        pr_body = f'She has a body mass index (BMI) of {np.clip(np.round(bmi,2), a_min=0, a_max=None)} kilograms per square meters' +\
        f' and triceps skin fold thickness of {np.clip(np.round(skin,2), a_min=0, a_max=None)} mm.'
        pr_insulin = f'Her two-hours serum insulin is {np.clip(np.round(insulin,2), a_min=0, a_max=None)} microunits per milliliter.'
        pr_diabetes = f'Her diabetes pedigree function is {np.clip(np.round(diabetes,2), a_min=0.08, a_max=2.42)}.'
        
        # pr_label = {0: 'This patient has diabetes.',
        #             1: 'This patient does not have diabetes.'
        #             }

        pr_label = {0: 'Does this patient have diabetes? Yes or No? Answer: No',
                    1: 'Does this patient have diabetes? Yes or No? Answer: Yes'
                    }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        prompt = 'The following describes the diagnostic measurements of a female patient of Pima Indian heritage.' +\
        f' {pr_age} {pr_pregnancy} {pr_glucose} {pr_bp} {pr_body} {pr_insulin} {pr_diabetes}'

        if training:
            training_prompt =  prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt +' Does this patient have diabetes? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def heart_sf(data, training=True):
    prompts = []
    testing_labels = []
    for index, row in data.iterrows():
        age, sex, chestpain, bp, cholestrol, fasting, ecg, hr, angina, oldpeak, slope, label = row
        
        if sex=='F':
            sex = 'female'
            pro = ['She', 'Her', 'Her', 'Hers'] 
        else:
            sex = 'male'
            pro = ['He', 'Him', 'His', 'His']
            
        pr_chestpain = {
            'ASY': f'{pro[0]} has asymptomatic chest pain.', 
            'TA': f'{pro[0]} has typical angina chest pain.', 
            'ATA': f'{pro[0]} has atypical angina chest pain.',  
            'NAP': f'{pro[0]} has non-anginal chest pain.'
        }
        
        pr_fasting = {
            0: f'{pro[0]} has fasting blood sugar lower than 120 milligrams per decilitre.',
            1: f'{pro[0]} has fasting blood sugar greater than 120 milligrams per decilitre.'
        }
        
        pr_ecg = {
            'Normal': f'{pro[2]} resting electrocardiogram results are normal.',
            'LVH': f'{pro[2]} resting electrocardiogram results show probable or definite left ventricular' +\
            ' hypertrophy by Estes\' criteria.',
            'ST': f'{pro[2]} resting electrocardiogram results have ST-T wave abnormality' +\
            ' (T wave inversions and/or ST elevation or depression greater than 0.05 minute volume).'
        }
        
        pr_angina = {
            'N': f'{pro[0]} does not have exercise induced angina.',
            'Y': f'{pro[0]} has exercise induced angina.'
        }
        
        pr_slope = {
            'Flat': f'The peak exercise ST segment for this patient is flat.',
            'Down': f'The peak exercise ST segment for this patient slopes upwards.',
            'Up': f'The peak exercise ST segment for this patient slopes downwards.'
        }
        
        # pr_label = {
        #     0: f'This patient does not have heart disease.',
        #     1: f'This patient has heart disease.'
        # }

        pr_label = {
            0: 'Does this patient have heart disease? Yes or No? Answer: No',
            1: 'Does this patient have heart disease? Yes or No? Answer: Yes'
        }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        pr_age = f'This patient is a {np.clip(round(age), a_min=0, a_max=100)} years old {sex}.'
        pr_bp = f'{pro[0]} has a resting blood pressure of {np.clip(np.round(bp,2), a_min=0, a_max=None)} mm Hg.'
        pr_cholestrol = f'{pro[2]} serum cholesterol is {np.clip(np.round(cholestrol,2), a_min=0, a_max=None)} milligrams per deciliter.'
        pr_hr = f'The maximum heart rate achieved for this patient is {np.clip(np.round(hr), a_min=0, a_max=None)}.'
        pr_oldpeak = f'{pro[2]} ST depression induced by exercise relative to rest is {np.round(oldpeak,2)}.'
        
        prompt = 'The following describes diagnostic measurements of a patient.' +\
        f' {pr_age} {pr_chestpain[chestpain]} {pr_bp} {pr_cholestrol} {pr_fasting[np.clip(round(fasting), a_min=0, a_max=1)]}' +\
        f' {pr_ecg[ecg]} {pr_hr} {pr_angina[angina]} {pr_oldpeak} {pr_slope[slope]}'
    
        if training:
            training_prompt =  prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt +' Does this patient have heart disease? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def jungle_sf(data, training=True):
    prompts = []
    testing_labels = []
    for index, row in data.iterrows():
        wstrength, wfile, wrank, bstrength, bfile, brank, label  = row

        # pr_label = {
        #     False: 'The white player does not win this game.',
        #     True: 'The white player wins this game.'
        # }
        
        pr_label = {
            0: 'Does the white player win this game? Yes or No? Answer: No',
            1: 'Does the white player win this game? Yes or No? Answer: Yes'
        }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        prompt = f'In a two-piece endgame of jungle chess, the white piece has strength {np.clip(round(wstrength), a_min=0, a_max=7)}' +\
        f' and is on file {np.clip(round(wfile), a_min=0, a_max=6)} and rank {np.clip(round(wrank), a_min=0, a_max=8)}. The black piece has strength' +\
        f' {np.clip(round(bstrength), a_min=0, a_max=7)} and is on file {np.clip(round(bfile), a_min=0, a_max=6)} and rank {np.clip(round(brank), a_min=0, a_max=8)}.'
        
        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Does the white player win this game? Yes or No?'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def groupings(name, data, k):
    if name == 'adult':
        return group_by_adult(data, k)
    elif name == 'bank':
        return group_by_bank(data, k)
    elif name == 'blood':
        return group_by_blood(data, k)
    elif name == 'calhousing':
        return group_by_calhousing(data, k)
    elif name == 'car':
        return group_by_car(data, k)
    elif name == 'diabetes':
        return group_by_diabetes(data, k)
    elif name == 'heart':
        return group_by_heart(data, k)
    elif name == 'jungle':
        return group_by_jungle(data, k)

def group_by_adult(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]
        
    # k = 4 label, sex
    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['sex'] == 'Male') & (df_sampled['label'] == True)]
        s_2 = df_sampled.loc[(df_sampled['sex'] == 'Male') & (df_sampled['label'] == False)]
        s_3 = df_sampled.loc[(df_sampled['sex'] == 'Female') & (df_sampled['label'] == True)]
        s_4 = df_sampled.loc[(df_sampled['sex'] == 'Female') & (df_sampled['label'] == False)]
        splits = [s_1, s_2, s_3, s_4]
   
    # k = 8 label, sex, race
    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['sex'] == 'Male') & (df_sampled['label'] == True) & (df_sampled['race'] == 'White')]
        s_2 = df_sampled.loc[(df_sampled['sex'] == 'Male') & (df_sampled['label'] == False) & (df_sampled['race'] == 'White')]
        s_3 = df_sampled.loc[(df_sampled['sex'] == 'Female') & (df_sampled['label'] == True) & (df_sampled['race'] == 'White')]
        s_4 = df_sampled.loc[(df_sampled['sex'] == 'Female') & (df_sampled['label'] == False) & (df_sampled['race'] == 'White')]
        s_5 = df_sampled.loc[(df_sampled['sex'] == 'Male') & (df_sampled['label'] == True) & (df_sampled['race'] != 'White')]
        s_6 = df_sampled.loc[(df_sampled['sex'] == 'Male') & (df_sampled['label'] == False) & (df_sampled['race'] != 'White')]
        s_7 = df_sampled.loc[(df_sampled['sex'] == 'Female') & (df_sampled['label'] == True) & (df_sampled['race'] != 'White')]
        s_8 = df_sampled.loc[(df_sampled['sex'] == 'Female') & (df_sampled['label'] == False) & (df_sampled['race'] != 'White')]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]
  
    return splits

def group_by_bank(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]
        
    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['poutcome'] == 'success')]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['poutcome'] == 'success')]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['poutcome'] != 'success')]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['poutcome'] != 'success')]
        splits = [s_1, s_2, s_3, s_4]

    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['poutcome'] == 'success') & (df_sampled['age'] <= 50)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['poutcome'] == 'success') & (df_sampled['age'] <= 50)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['poutcome'] != 'success') & (df_sampled['age'] <= 50)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['poutcome'] != 'success') & (df_sampled['age'] <= 50)]
        s_5 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['poutcome'] == 'success') & (df_sampled['age'] > 50)]
        s_6 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['poutcome'] == 'success') & (df_sampled['age'] > 50)]
        s_7 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['poutcome'] != 'success') & (df_sampled['age'] > 50)]
        s_8 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['poutcome'] != 'success') & (df_sampled['age'] > 50)]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]
        
    return splits

def group_by_blood(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]

    # k = 4 label, frequency
    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['frequency'] <= 5)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['frequency'] <= 5)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['frequency'] > 5)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['frequency'] > 5)]
        splits = [s_1, s_2, s_3, s_4]

    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['frequency'] <= 5) & (df_sampled['recency'] <= 10)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['frequency'] <= 5) & (df_sampled['recency'] <= 10)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['frequency'] > 5) & (df_sampled['recency'] <= 10)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['frequency'] > 5) & (df_sampled['recency'] <= 10)]
        s_5 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['frequency'] <= 5) & (df_sampled['recency'] > 10)]
        s_6 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['frequency'] <= 5) & (df_sampled['recency'] > 10)]
        s_7 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['frequency'] > 5) & (df_sampled['recency'] > 10)]
        s_8 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['frequency'] > 5) & (df_sampled['recency'] > 10)]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]

    return splits

def group_by_calhousing(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]

    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['housing_median_age'] <= 25)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['housing_median_age'] <= 25)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['housing_median_age'] > 25)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['housing_median_age'] > 25)]
       
        splits = [s_1, s_2, s_3, s_4]

    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['housing_median_age'] <= 25) & (df_sampled['population'] <= 2000)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['housing_median_age'] <= 25) & (df_sampled['population'] <= 2000)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['housing_median_age'] > 25) & (df_sampled['population'] <= 2000)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['housing_median_age'] > 25) & (df_sampled['population'] <= 2000)]
        s_5 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['housing_median_age'] <= 25) & (df_sampled['population'] > 2000)]
        s_6 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['housing_median_age'] <= 25) & (df_sampled['population'] > 2000)]
        s_7 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['housing_median_age'] > 25) & (df_sampled['population'] > 2000)]
        s_8 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['housing_median_age'] > 25) & (df_sampled['population'] > 2000)]
       
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]

    return splits

def group_by_car(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[(df_sampled['label'] == 0)]
        s_2 = df_sampled.loc[(df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)]
        splits = [s_1, s_2]
    
    # k = 2 label (False, True), buying
    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == 0) & ((df_sampled['buying'] == 'low') | (df_sampled['buying'] == 'med'))]
        s_2 = df_sampled.loc[((df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)) & ((df_sampled['buying'] == 'low') | (df_sampled['buying'] == 'med'))]
        s_3 = df_sampled.loc[(df_sampled['label'] == 0) & ((df_sampled['buying'] == 'high') | (df_sampled['buying'] == 'vhigh'))]
        s_4 = df_sampled.loc[((df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)) & ((df_sampled['buying'] == 'high') | (df_sampled['buying'] == 'vhigh'))]
        splits = [s_1, s_2, s_3, s_4]

    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == 0) & ((df_sampled['buying'] == 'low') | (df_sampled['buying'] == 'med')) & ((df_sampled['doors'] == '2') | (df_sampled['buying'] == '3'))]
        s_2 = df_sampled.loc[((df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)) & ((df_sampled['buying'] == 'low') | (df_sampled['buying'] == 'med')) & ((df_sampled['doors'] == '2') | (df_sampled['buying'] == '3'))]
        s_3 = df_sampled.loc[(df_sampled['label'] == 0) & ((df_sampled['buying'] == 'high') | (df_sampled['buying'] == 'vhigh')) & ((df_sampled['doors'] == '2') | (df_sampled['buying'] == '3'))]
        s_4 = df_sampled.loc[((df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)) & ((df_sampled['buying'] == 'high') | (df_sampled['buying'] == 'vhigh')) & ((df_sampled['doors'] == '2') | (df_sampled['buying'] == '3'))]
        s_5 = df_sampled.loc[(df_sampled['label'] == 0) & ((df_sampled['buying'] == 'low') | (df_sampled['buying'] == 'med')) & ((df_sampled['doors'] == '4') | (df_sampled['buying'] == '5more'))]
        s_6 = df_sampled.loc[((df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)) & ((df_sampled['buying'] == 'low') | (df_sampled['buying'] == 'med')) & ((df_sampled['doors'] == '4') | (df_sampled['buying'] == '5more'))]
        s_7 = df_sampled.loc[(df_sampled['label'] == 0) & ((df_sampled['buying'] == 'high') | (df_sampled['buying'] == 'vhigh')) & ((df_sampled['doors'] == '4') | (df_sampled['buying'] == '5more'))]
        s_8 = df_sampled.loc[((df_sampled['label'] == 1) | (df_sampled['label'] == 2) | (df_sampled['label'] == 3)) & ((df_sampled['buying'] == 'high') | (df_sampled['buying'] == 'vhigh')) & ((df_sampled['doors'] == '4') | (df_sampled['buying'] == '5more'))]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]
        
    return splits

def group_by_diabetes(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]

    # k = 4 label (False, True), pregnanices
    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Pregnancies'] <= 4)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Pregnancies'] <= 4)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Pregnancies'] > 4)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Pregnancies'] > 4)]
        splits = [s_1, s_2, s_3, s_4]
    
    # label, pregnancies, age
    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Pregnancies'] <= 4) & (df_sampled['Age'] <= 33)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Pregnancies'] <= 4) & (df_sampled['Age'] <= 33)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Pregnancies'] > 4) & (df_sampled['Age'] <= 33)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Pregnancies'] > 4) & (df_sampled['Age'] <= 33)]
        s_5 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Pregnancies'] <= 4) & (df_sampled['Age'] > 33)]
        s_6 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Pregnancies'] <= 4) & (df_sampled['Age'] > 33)]
        s_7 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Pregnancies'] > 4) & (df_sampled['Age'] > 33)]
        s_8 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Pregnancies'] > 4) & (df_sampled['Age'] > 33)]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]

    return splits

def group_by_heart(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]

    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Sex']=='F')]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Sex']=='F')]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Sex']=='M')]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Sex']=='M')]
        splits = [s_1, s_2, s_3, s_4]

    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Sex']=='F') & (df_sampled['ExerciseAngina']=='N')]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Sex']=='F') & (df_sampled['ExerciseAngina']=='N')]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Sex']=='M') & (df_sampled['ExerciseAngina']=='N')]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Sex']=='M') & (df_sampled['ExerciseAngina']=='N')]
        s_5 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Sex']=='F') & (df_sampled['ExerciseAngina']=='Y')]
        s_6= df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Sex']=='F') & (df_sampled['ExerciseAngina']=='Y')]
        s_7 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['Sex']=='M') & (df_sampled['ExerciseAngina']=='Y')]
        s_8 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['Sex']=='M') & (df_sampled['ExerciseAngina']=='Y')]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]

    return splits

def group_by_jungle(df_sampled, k):
    splits = []
    
    # k = 1 whole s
    if k == 1:
        return [df_sampled]
        
    # k = 2 label (False, True)
    elif k == 2:
        s_1 = df_sampled.loc[df_sampled['label'] == True]
        s_2 = df_sampled.loc[df_sampled['label'] == False]
        splits = [s_1, s_2]

    elif k == 4:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['white_piece0_strength'] <= 4)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['white_piece0_strength'] <= 4)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['white_piece0_strength'] > 4)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['white_piece0_strength'] > 4)]
        splits = [s_1, s_2, s_3, s_4]

    elif k == 8:
        s_1 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['white_piece0_strength'] <= 4) & (df_sampled['black_piece0_strength'] <= 4)]
        s_2 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['white_piece0_strength'] <= 4) & (df_sampled['black_piece0_strength'] <= 4)]
        s_3 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['white_piece0_strength'] > 4) & (df_sampled['black_piece0_strength'] <= 4)]
        s_4 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['white_piece0_strength'] > 4) & (df_sampled['black_piece0_strength'] <= 4)]
        s_5 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['white_piece0_strength'] <= 4) & (df_sampled['black_piece0_strength'] > 4)]
        s_6 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['white_piece0_strength'] <= 4) & (df_sampled['black_piece0_strength'] > 4)]
        s_7 = df_sampled.loc[(df_sampled['label'] == True) & (df_sampled['white_piece0_strength'] > 4) & (df_sampled['black_piece0_strength'] > 4)]
        s_8 = df_sampled.loc[(df_sampled['label'] == False) & (df_sampled['white_piece0_strength'] > 4) & (df_sampled['black_piece0_strength'] > 4)]
        splits = [s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]

    return splits

"""
Categorical
"""
def categorical(feat, epsilon):
    val_counts = {}

    if len(feat.unique()) == 1:
        return feat.unique()[0]
        
    for val, count in feat.value_counts().items():
        val_counts[val] = count + np.random.laplace(loc=0, scale=1/epsilon)
    dp_avg = max(val_counts, key=val_counts.get)
    
    return dp_avg

"""
Categorical
"""
def categorical_no_dp(feat):
    val_counts = {}

    if len(feat.unique()) == 1:
        return feat.unique()[0]
        
    for val, count in feat.value_counts().items():
        val_counts[val] = count 
    dp_avg = max(val_counts, key=val_counts.get)
    
    return dp_avg

"""
Numerical
"""
def numerical(feat, high, low, epsilon):
    dp_count = len(feat) + np.random.laplace(loc=0, scale=1/(epsilon/2))
    dp_sum = feat.sum() + np.random.laplace(loc=0, scale=(high-low)/(epsilon/2))
    return dp_sum / dp_count

"""
Numerical
"""
def numerical_no_dp(feat):
    dp_count = len(feat) 
    dp_sum = feat.sum()
    return dp_sum / dp_count

def get_high(name):
    if name == 'adult':
        return [100, 0, 0, 0, 0, 0, 0, 0, 0, 50000, 5000, 40, 0, 1]
    elif name == 'bank':
        return [100, 0, 0, 0, 0, 50000, 0, 0, 0, 31, 0, 5000, 63, 900, 300, 0, 1]
    elif name == 'blood':
        return [120, 60, 30000, 120, 1]  #[10 years, 6 (num times can a year) * 10 years, 500cc(whats wanted) * 6 max times per year * 10 years), 10 years0]
    elif name == 'calhousing':
        return [0, 60000, 32.2, 3600, 1800, 1800, 600, 42, -115] #[median california income 1990, median housing age national, 6 rooms * 600 houses in a block, 3 bedrooms * 600 houses, 3 people * 600 houses, 600 houses
    elif name == 'car':
        return [0,0,0,0,0,0,4]
    elif name == 'diabetes':
        return [19, 200, 140, 100, 100, 600, 100, 2.42, 100] #[19 kids and counting, 200 means diabetes, threshold for high blood pressure, v high skin thickness]
    elif name == 'heart':
        return [100, 0, 0, 200, 500, 1, 0, 210, 0, 4, 0, 0]
    elif name == 'jungle':
        return [7, 6, 8, 7, 6, 8]

def get_low(name):
    if name == 'adult':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif name == 'bank':
        return [0, 0, 0, 0, 0, -5000, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    elif name == 'blood':
        return [0, 0, 2, 2, 0] # 2 - minimum number of months between donations
    elif name == 'calhousing':
        return [0, 0, 0, 1, 1, 1, 1, 32, -124]
    elif name == 'car':
        return [0,0,0,0,0,0,0]
    elif name == 'diabetes':
        return [0, 0, 0, 0, 0, 0, 0, 0, .08, 21]
    elif name == 'heart':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0 ]
    elif name == 'jungle':
        return [0, 0, 0, 0, 0, 0]

"""
Main Process
"""
def get_gdp(dataset, n, k, eps, set):
    possible = ['adult', 'bank', 'blood', 'calhousing', 'car', 'diabetes', 'heart', 'jungle']
    if dataset not in possible:
        print("Please select correct dataset name: adult, bank, blood, calhousing, car, diabetes, heart, jungle")
        exit()
    train_filename = f'{dataset}_original.csv'
    df = pd.read_csv(DATA_DIR / dataset/ train_filename)
    df = df.dropna(axis=0)

    if dataset == 'diabetes' or dataset == 'heart':
        df['label'] = df['label'].astype('bool')
    epsilon = eps / len(df.columns)

    if eps:
        high = get_high(dataset)
        low = get_low(dataset)
    
        # sample dfTrain according to Pois(n / N)
        items_to_sample = [(np.random.poisson(n / len(df)) >= 1) for i in range(len(df))]
        df_sampled = df[items_to_sample]

         # split into k partitions based on Group By
        splits = groupings(dataset, df_sampled, k)

        if len(splits) < 1:
            print("Select correct number of splits: 1, 2, 4, 8")
            exit()
        # perform DP average for each split
        # for each feature & label
            # take numerical or categorical DP avg
        all_avg = []
        for s in splits:
            dp_avg = []
            for i, f in enumerate(s.columns):
                if s[f].dtype != 'object' and s[f].dtype != 'bool':
                    dp_avg_f = round(numerical(s[f], high[i], low[i], epsilon),2)
                else:
                    dp_avg_f = categorical(s[f], epsilon)
                dp_avg.append(dp_avg_f)
            all_avg.append(dp_avg)
    else:
    # for no dp 
        df_sampled = df.sample(n=n, replace=False)
        
        # split into k partitions based on Group By
        splits = groupings(dataset, df_sampled, k)
        if len(splits) < 1:
            print("Select correct number of splits: 1, 2, 4, 8")
            exit()
        all_avg = []
        for s in splits:
            avg = []
            for i, f in enumerate(s.columns):
                if s[f].dtype != 'object' and s[f].dtype != 'bool':
                    avg_f = numerical_no_dp(s[f])
                else:
                    avg_f = categorical_no_dp(s[f])
                avg.append(avg_f)
            all_avg.append(avg)
        
    # turn into dataframe
    dp_df = pd.DataFrame(all_avg, columns = df.columns)
    
    # seralize 
    demonstration, _ = seralize(dataset, dp_df, True)
    
    # add test examples and demonstration examples to dict 
    test_filename = f'{dataset}_original_test.csv'
    df_test = pd.read_csv(DATA_DIR / dataset/ test_filename)
    df_test = df_test.dropna(axis=0)

    large = ['adult', 'bank', 'calhousing', 'jungle']
    
    if dataset in large:
        df_test_sample = df_test.sample(n=int(len(df_test)*.1))
    else: 
        df_test_sample = df_test.sample(n=int(len(df_test)*.75))
        
    test_demonstrations, test_labels = seralize(dataset, df_test_sample, False)
    
    prompts = {}
    for i, (q, y) in enumerate(zip(test_demonstrations, test_labels)):
        prompts[i] = {'demonstration': demonstration, 'query': q, 'label': y}

    save_file_name = f'gdp-files/{dataset}/eps-{eps}/{k}/{dataset}-n-{n}-k-{k}-eps-{eps}-num-{set}-gdp.pkl'
    Path(f'gdp-files/{dataset}/eps-{eps}/{k}/').mkdir(parents=True, exist_ok=True)
    with open(save_file_name, 'wb') as handle:
        pickle.dump(prompts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(dataset_name, sample_size, eps, k, rounds):
    for i in range(rounds):
        get_gdp(dataset_name, sample_size, k, eps, i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', dest="dataset", help="Choice from: adult, bank, blood, calhousing, car, diabetes, heart, jungle.", type=str)
    parser.add_argument('--e', dest="epsilon", help="Choice of epsilon. Takes one int value or None", type=int)
    parser.add_argument('--r', dest="rounds", help="Number of training files to generate", type=int)
    parser.add_argument('--k', dest="num_splits", help="Number of splits to perform: 1, 2, 4, 8", type=int)
    parser.add_argument('--s', dest="sample_size", help="Size of subset S", type=int)
    args = parser.parse_args()
    main(args.dataset, args.sample_size, args.epsilon, args.num_splits, args.rounds)
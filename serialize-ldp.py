import math
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

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
        print("Please choose dataset from: adult, bank, blood, calhousing, car, diabetes, heart, jungle")
        exit()

def adult_sf(data, training=True):
    prompts = []
    testing_labels= []
        
    for index, row in data.iterrows():
        age, workclass, marital, race, sex, gain, hours, label, assoc, college, hs, prof_sch, less_hs = row
        
        cmpr = {0: 'less than', 1: 'more than'}
    
        if sex is False:
            sex = 'female'
            pro = ['She', 'Her', 'Her', 'Hers'] 
        else:
            sex = 'male'
            pro = ['He', 'Him', 'His', 'His']
        
        pr_workclass = {
            True: f'{pro[0]} works in the private sector.',
            False: f'{pro[0]} does not work in the private sector.'
        }
        
        if assoc == 1:
            pr_education = f'{pro[0]} has an associate\'s degree at most.'
        elif college==1:
            pr_education = f'{pro[0]} has a college degree at most.'
        elif hs == 1:
            pr_education = f'{pro[0]} has a high school degree at most.'
        elif prof_sch == 1:
            pr_education = f'{pro[0]} has attended a professional school at most.'
        else:
            pr_education = f'{pro[0]} has not completed high school.'

        pr_marital = {
            True: f'{pro[0]} is married.',
            False: f'{pro[0]} is not married.'
        }
            
        pr_race = {
            True: f'{pro[0]} is White.',
            False: f'{pro[0]} is not White.'
        }
        
        pr_age_sex = {
            0: f'This person is a {sex.lower()} less than 39 years of age.',
            1: f'This person is a {sex.lower()} 39 or more years of age.'
        }
        
        pr_hours = {
            0: f'{pro[0]} works less than 40 hours per week.',
            1: f'{pro[0]} works 40 or more hours per week.'
        }
        
        pr_capital = f'{pro[2]} capital gain was {cmpr[gain]} $1092 last year.' 

        if label == True:
            pr_label = 'Does this person earn more than $50,000 dollars annually? Yes or No? Answer: Yes'
        else:
            pr_label = 'Does this person earn more than $50,000 dollars annually? Yes or No? Answer: No'

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        # Get prompt strings
        prompt = 'An individual recorded in the 1994 US census is described as follows:' +\
        f' {pr_age_sex[age]} {pr_education} {pr_workclass[workclass]}' +\
        f' {pr_hours[hours]} {pr_capital} {pr_race[race]}' +\
        f' {pr_marital[marital]}'

        if training:
            training_prompt = prompt + ' ' + pr_label
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Does this person earn more than $50,000 dollars annually? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)
            
    return prompts, testing_labels

def bank_sf(data, training=True):
    prompts = []
    testing_labels= []
    
    for index, row in data.iterrows():
        age, balance, housing, day, duration, campaign, pdays, previous, poutcome, label  = row

        pr_age = {
            0: 'The client is 40 or less years of age.',
            1: 'The client is more than 40 years of age.'
        }
        
        pr_balance = {
            0: 'Their average yearly balance is less than 1362 euros.',
            1: 'Their average yearly balance is more than 1362 euros.',
        }
        
        pr_housing = {
            True: 'They have housing loans.',
            False: 'They do not have any housing loans.'
        }
        
        pr_duration = {
            0: 'and were last contacted for less than 258 seconds.',
            1: 'and were last contacted for more than 258 seconds.'
        }
        
        pr_campaign = {
            0: 'They were contacted less than 3 times during this campaign',
            1: 'They were contacted 3 or more times during this campaign'
        }
        
        if previous == 0:
            pr_previous = 'They were not contacted in previous campaigns.'
        else:
            if pdays == 0:
                pr_previous = f'They were contacted one or more times in a previous campaign and were last contacted less than 40 days ago for the previous campaign.'
            else:
                pr_previous = f'They were contacted one or more times in a previous campaign and were last contacted more than 40 days ago for the previous campaign.'
            
        pr_outcome = {
            True: 'The outcome of the previous marketing campaign was success for this client.',
            False: 'The outcome of the previous marketing campaign was either failure or unknown for this client.'
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
        
            
        prompt = 'A client at a Portuguese banking institution is described as follows:' +\
        f' {pr_age[age]} {pr_balance[balance]} {pr_housing[housing]} {pr_campaign[campaign]} {pr_duration[duration]} {pr_previous} {pr_outcome[poutcome]}'

        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Does this person subscribe to a term deposit? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)
            
    return prompts, testing_labels

def blood_sf(data, training=True):
    prompts = []
    testing_labels= []
    for index, row in data.iterrows():
        recency, frequency, monetray, time, label = row

        pr_frequency = {
            0: f'The donor has donated blood less than 6 times.',
            1: f'The donor has donated blood 6 or more times.'
        }
        
        pr_monetray = {
            0: f'In total, they have donated less than 1379 c.c. of blood.',
            1: f'In total, they have donated more than 1379 c.c. of blood.'
        }
        
        pr_recency = {
            0: f'They last donated blood less than 10 months ago.',
            1: f'They last donated blood 10 or more months ago.'
        }
        
        pr_time = {
            0: f'Their first blood donation was less than 34 months ago.',
            1: f'Their first blood donation was more than 34 months ago.'
        }

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
        f' {pr_frequency[frequency]} {pr_monetray[monetray]} {pr_recency[recency]} {pr_time[time]}'

        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Did the donor donate blood in March 2007? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def calhousing_sf(data, training=True):
    prompts = []
    testing_labels= []
    for index, row in data.iterrows():
        label, income, age, rooms, bedrooms, population, households, latitude, longitude = row

        cmpr = {0: 'less than', 1: 'more than'}

        pr_loc = f'This housing block is located at latitude {cmpr[latitude]} 36 and longitude {cmpr[longitude]} -120.'
        pr_rooms = f'The houses in the block have {cmpr[rooms]} 2636 rooms with {cmpr[bedrooms]} 538 bedrooms in total.'
        pr_median = f'The median age of houses in the block is {cmpr[age]} 29 years.'
        pr_households = f'There are {cmpr[households]} 500 total households in the block' +\
        f' with a total population {cmpr[population]} 1425.'
        pr_income = f'The median income of the households in the block is {cmpr[income]} 40,000 dollars.'
        
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
            testing_prompt = prompt + ' Is this housing block valuable? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def car_sf(data, training=True):
    prompts = []
    testing_labels = []
    
    for index, row in data.iterrows():
        persons, lug_boot, label, buy_h, buy_l, buy_m, maint_h, maint_l, maint_m, safety_h, safety_l, safety_m = row
    
        if buy_l == 1:
            pr_buying = f'The buying price of this car is low.'
        elif buy_m == 1:
            pr_buying = f'The buying price of this car is medium.'
        elif buy_h == 1:
            pr_buying = f'The buying price of this car is high.'
        else:
            pr_buying = f'The buying price of this car is unknown'
        
        if maint_l == 1:
            pr_maint = f'The maintenance cost for this car is low.'
        elif maint_m == 1:
            pr_maint = f'The maintenance cost for this car is medium.'
        elif maint_h == 1:
            pr_maint = f'The maintenance cost for this car is high.'
        else:
            pr_maint = f'The maintenance cost for this car is unkown.'
        
            
        if safety_l == 1:
            pr_safety = 'The safety rating of this car is estimated to be low.'
        elif safety_m == 1:
            pr_safety = 'The safety rating of this car is estimated to be medium.'
        elif safety_h == 1:
            pr_safety = 'The safety rating of this car is estimated to be high.'
        else:
            pr_safety = 'The safety rating of this car is unknown.'
        
        pr_persons = {
            0: 'The car can 2 people.',
            1: 'The car can fit 4 or more people.'
        }
        
        pr_lug = {
            0: 'The luggage boot in this car is small.',
            1: 'The luggage boot in this car is medium-sized or big.'
        }

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
        f' {pr_buying} {pr_maint} {pr_persons[persons]} {pr_lug[lug_boot]} {pr_safety}'
       
        if training:
            training_prompt = prompt + ' ' + pr_label[np.clip(round(label), a_min=0, a_max=3)]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Is this car acceptable? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label[label])

    return prompts, testing_labels

def diabetes_sf(data, training=True):
    prompts = []
    testing_labels = []
    for index, row in data.iterrows():
        pregnancy, glucose, bp, skin, insulin, bmi, diabetes, age, label = row
        
        # Context: In particular, all patients here are females at least 21 years old of Pima Indian heritage.

        cmpr = {0: 'less than', 1: 'more than'}

        pr_age = {
            0: f'This patient is less than 34 years of age.',
            1: f'This patient is 34 or more years of age.'
        }
        
        pr_pregnancy = {
            0: f'She has been pregnant less than 4 times.',
            1: f'She has been pregnant 4 or more times.'
        }
        
        pr_glucose = f'Her plasma glucose concentration at two hours in an oral glucose tolerance test' +\
        f' is {cmpr[glucose]} 121 milligrams per deciliter.'
        pr_bp = f'Her blood pressure is measured to be {cmpr[bp]} 69 mm Hg.'
        pr_body = f'She has a body mass index (BMI) of {cmpr[bmi]} 32 kilograms per square meters' +\
        f' and triceps skin fold thickness of {cmpr[skin]} 21 mm.'
        pr_insulin = f'Her two-hours serum insulin is {cmpr[insulin]} 80 microunits per milliliter.'
        pr_diabetes = f'Her diabetes pedigree function is {cmpr[diabetes]} 0.5.'

        pr_label = {0: 'Does this patient have diabetes? Yes or No? Answer: No',
                    1: 'Does this patient have diabetes? Yes or No? Answer: Yes'
                    }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        prompt = 'The following describes the diagnostic measurements of a female patient of Pima Indian heritage.' +\
        f' {pr_age[age]} {pr_pregnancy[pregnancy]} {pr_glucose} {pr_bp} {pr_body} {pr_insulin} {pr_diabetes}'

        if training:
            training_prompt =  prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt +' Does this patient have diabetes? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

def heart_sf(data, training=True):
    prompts = []
    testing_labels = []
    for index, row in data.iterrows():
        age, sex, cpt, bp, cholestrol, fasting, ecg, hr, angina, oldpeak, slope, label = row
        
        cmpr = {0: 'less than', 1: 'more than'}

        if sex==0:
            sex = 'female'
            pro = ['She', 'Her', 'Her', 'Hers'] 
        else:
            sex = 'male'
            pro = ['He', 'Him', 'His', 'His']
        
        pr_age = {
            0: f'This patient is {sex} and less than 54 years of age.',
            1: f'This patient is {sex} and 54 or more years of age.'
        }
        
        if cpt == 1:
            pr_chestpain = f'{pro[0]} has angina chest pain.'
        else:
            pr_chestpain = f'{pro[0]} does not have angina chest pain.'
        
        pr_fasting = {
            0: f'{pro[2]} fasting blood sugar is lower than 120 milligrams per deciliter.',
            1: f'{pro[2]} fasting blood sugar is greater than 120 milligrams per deciliter.'
        }
        
        pr_ecg = {
            0 : f'{pro[2]} resting electrocardiogram results are normal.',
            1: f'{pro[2]} resting electrocardiogram results show probable or definite left ventricular hypertrophy by Estes\' criteria or have ST-T wave abnormality (T wave inversions and/or ST elevation or depression greater than 0.05 minute volume).'
        }
         
        pr_angina = {
            0: f'{pro[0]} does not have exercise induced angina.',
            1: f'{pro[0]} has exercise induced angina.'
        }
        
        pr_slope = {
            0: f'The peak exercise ST segment for this patient is flat.',
            1: f'The peak exercise ST segment for this patient has a slope.'
        }

        pr_label = {
            0: 'Does this patient have heart disease? Yes or No? Answer: No',
            1: 'Does this patient have heart disease? Yes or No? Answer: Yes'
        }

        if training == False:
            if label == True:
                pr_label = 'Yes'
            else:
                pr_label = 'No'
                
        pr_bp = f'{pro[0]} has a resting blood pressure of {cmpr[bp]} 132 mm Hg.'
        pr_cholestrol = f'{pro[2]} serum cholestrol is {cmpr[cholestrol]} 199 milligrams per deciliter.'
        pr_hr = f'The maximum heart rate achieved for this patient is {cmpr[hr]} 137.'
        pr_oldpeak = f'{pro[2]} ST depression induced by exercise relative to rest is {cmpr[oldpeak]} 0.9.'
        
        prompt = 'The following describes diagnostic measurements of a patient.' +\
        f' {pr_age[age]} {pr_chestpain} {pr_bp} {pr_cholestrol} {pr_fasting[fasting]}' +\
        f' {pr_ecg[ecg]} {pr_hr} {pr_angina[angina]} {pr_oldpeak} {pr_slope[slope]}'
    
        if training:
            training_prompt =  prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt +' Does this patient have heart disease? Yes or No? Answer:'
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
                
        pr_wstrength = {
            0: f'the white piece has strength less than 4.2',
            1: f'the white piece has strength more than 4.2'
        }
        
        pr_wfile = {
            0: f'and is on file less than 3',
            1: f'and is on file 3 or more'
        }
        
        pr_wrank = {
            0: f'and rank less than 4',
            1: f'and rank 3 or more'
        }
        
        pr_bstrength = {
            0: f'The black piece has strength less than 4.2',
            1: f'The black piece has strength more than 4.2'
        }
        
        pr_bfile = {
            0: f'and is on file less than 3',
            1: f'and is on file 3 or more'
        }
        
        pr_brank = {
            0: f'and rank less than 4',
            1: f'and rank 3 or more'
        }

        prompt = f'In a two-piece endgame of jungle chess, {pr_wstrength[wstrength]} {pr_wfile[wfile]}' +\
        f' {pr_wrank[wrank]}. {pr_bstrength[bstrength]} {pr_bfile[bfile]} {pr_brank[brank]}.'
        
        if training:
            training_prompt = prompt + ' ' + pr_label[label]
            prompts.append(training_prompt)
        else:
            testing_prompt = prompt + ' Does the white player win this game? Yes or No? Answer:'
            prompts.append(testing_prompt)
            testing_labels.append(pr_label)

    return prompts, testing_labels

"""
Main Process
"""

def serialize_ldp(dataset_name, k, eps, r):
    if eps:
        train_filename = f'ldp-non-serial/{dataset_name}_ldp_eps_{eps}_train_{r}.csv'
    else:
        train_filename = f'ldp-non-serial/{dataset_name}_ldp_nodp_train_{r}.csv'
   
    df = pd.read_csv(train_filename)
    df = df.dropna(axis=0)

    y_train = df['label']
    X_train = df.drop(columns=['label'])

    # add test examples and demonstration examples to dict 
    test_filename = f'ldp-non-serial/{dataset_name}_ldp_eps_{eps}_test_{r}.csv'
    df_test = pd.read_csv(test_filename)
    df_test = df_test.dropna(axis=0)

    large = ['adult', 'bank', 'calhousing', 'jungle']
    
    if dataset_name in large:
        df_test_sample = df_test.sample(n=int(len(df_test)*.1))
    else: 
        df_test_sample = df_test.sample(n=int(len(df_test)*.75))

    test_demonstrations, test_labels = seralize(dataset_name, df_test_sample, False)
    
    prompts = {}
    count = 0
    for (i, row), q, y in zip(df_test_sample.iterrows(), test_demonstrations, test_labels):
        y_row = row['label']
        X_row = row.drop('label')

        X = X_row.values.reshape(1,-1)

        # sample dfTrain to get k
        df_sampled = df.sample(n=k, replace=False)
        
        # seralize 
        demonstration, _ = seralize(dataset_name, df_sampled, True)
        prompts[count] = {'demonstration': demonstration, 'query': q, 'label': y}
        count += 1
        
    Path(f'ldp-files/{dataset_name}/eps-{eps}/{k}/').mkdir(parents=True, exist_ok=True)
    save_file_name = f'ldp-files/{dataset_name}/eps-{eps}/{k}/{dataset_name}-k-{k}-eps-{eps}-num-{r}-ldp.pkl'
    with open(save_file_name, 'wb') as handle:
        pickle.dump(prompts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(dataset_name, eps, k, rounds):
    for i in range(rounds):
        serialize_ldp(dataset_name, k, eps, i+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', dest="dataset", help="Choice from: adult, bank, blood, calhousing, car, diabetes, heart, jungle.", type=str)
    parser.add_argument('--e', dest="epsilon", help="Choice of epsilon. Takes one int value or None", type=int)
    parser.add_argument('--r', dest="rounds", help="Number of training files to generate", type=int)
    parser.add_argument('--k', dest="num_splits", help="Number of splits to perform: 1, 2, 4, 8", type=int)
    args = parser.parse_args()
    main(args.dataset, args.epsilon, args.num_splits, args.rounds)

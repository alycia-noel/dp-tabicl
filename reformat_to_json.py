import glob
import pickle
import json
import argparse
from pathlib import Path

def reformat_prompts(loaded_data):
    prompt_list = []
    for prompt in loaded_data.keys():
        dic = {}
        dic['prompt'] = ''
        for i in range(len(loaded_data[prompt]['demonstration'])):
            if i==0:
                dic['prompt'] = loaded_data[prompt]['demonstration'][i]
            else:
                dic['prompt']= dic['prompt'] + '\n' + loaded_data[prompt]['demonstration'][i]
        dic['prompt'] = dic['prompt'] + '\n' + "{text} Answer:"
        dic['completion'] = '{label_word}'
        dic['label_words'] = ["Yes", "No"]
        dic['text'] = loaded_data[prompt]['query']
        dic['ground_truth'] = loaded_data[prompt]['label']
        prompt_list.append(dic)
    return prompt_list

#Load file
def main(which_method):
    files = glob.glob(f'{which_method}-files/**/*.pkl', recursive = True) 
    Path(f'{which_method}_json').mkdir(parents=True, exist_ok=True)
    
    for f in files:
        output_file = f'{which_method}_json/' + (f.split("/")[-1]).split(".")[0]+'.json'
        with open(f, 'rb') as handle:
            b = pickle.load(handle)
        prompts = reformat_prompts(b)
        with open(output_file, 'w') as fp:
            json.dump(prompts, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', dest="which_type", help="Choice from: ldp or gdp.", type=str)
    args = parser.parse_args()
    main(args.which_type)
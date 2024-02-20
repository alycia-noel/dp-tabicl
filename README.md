# DP-TabICL
This repository contains the code to run the experiments for _DP-TabICL: In-Context Learning with Differentially Private Tabular Data_ published on ArXiV and submitted to the COLM conferece. 

![dp-tabicl](https://github.com/alycia-noel/dp-tabicl/assets/71036958/e0141c44-f0ea-42aa-81e9-36ca7b47eb41)

### Requirements 
The code in this repository requires the Llama-2 model from Meta to be installed. Follow the instructions [here](https://github.com/facebookresearch/llama) to install the wanted Llama model. All code has been tested with both the Llama-2-7B and Llama-2-13B models. 

To install all additional requirements: 

```
pip install -e .
```

### LDP Experiments
To run the LDP experiments first the data files need to be generated. These files will contain the LDP reconstructed data from which examples can be pulled from to act as differentially private demonstrations.

```
python generate-ldp-data.py --d [dataset-name] --e [epsilon] --r [num-rounds]
```
where dataset-name can be chosen from [adult, bank, blood, calhousing, car, diabetes, heart, jungle], epsilon can be any integer or None (in our experiments we used 1, 2, 10, 25, 50) and num-rounds are the number of files to generate per setting (in our experiments we used r=5).

After generating the LDP protected data files, we then have to create the prompts to query the Llama model:
```
python serialize-ldp.py --d [dataset-name] --e [epsilon] --r [num-rounds] --k [num-shots]
```
Here, epsilon and num-rounds must be the same as values previously used to generate the LDP data files. Num-shots is the number of demostration examples used in a single prompt. In our experimentation we used k=1, 2, 4, and 8. 

To query the Llama model with the generated prompts, first we have to change the generated prompt files from pickle files to json:
```
python reformat_to_json.py --w ldp
```
Then, we can query the model on all generated prompts:

```
python run_dptabicl.py --i [input_folder] --o [output_text_file] --d [output_dir]
```
Here, the input_folder is the destination of the folder generated using the reformat_to_json.py script, the output_text_file is the wanted location of the text file that will contain the accuracy, F1 and timing reports for each json file, and output_dir is the wanted output directory for the detailed classification reports for each json file. 

It may be necessary to edit run_dptabicl.py to point to your specific installation location for the downloaded Llama model. Simply edit the --ckpt_dir and --tokenizer_path values in the commands list on line 20. 

### GDP Experiments
The GDP experiments are similar to the LDP, however, data generation and serialization are done in tandem therefore you only need to call 
```
python generate-gdp-data.py --d [dataset-name] --e [epsilon] --r [num-rounds]
```
before running
```
python reformat_to_json.py --w gdp
```
and 
```
python run_dptabicl.py --i [input_folder] --o [output_text_file] --d [output_dir]
```

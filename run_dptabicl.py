import glob
import shlex
import subprocess
import argparse
from pathlib import Path

def main(input_folder, output_text_file, output_dir):
    print('Starting')
    with open(output_text_file, 'w') as f:
            f.write('')
        
    files = glob.glob(f'{input_folder}/*.json', recursive = True)
    files.sort()
    Path(f'output_dir').mkdir(parents=True, exist_ok=True)
    for f in files:
        print('Working on file: ', (f.split("/")[-1]).split(".")[0]+'.json')
        output_file = f'{output_dir}' + (f.split("/")[-1]).split(".")[0]+'-results.json'
     
        #Add multiple commands for different inputs
        commands = [f'torchrun --nproc_per_node 2 --rdzv-endpoint localhost:29499 dp-tabicl.py --data_path {f} --output_path {output_file} --output_text_file {output_text_file} --ckpt_dir llama --tokenizer_path tokenizer.model',
                    ]
        for command in commands:
            result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE)
            print(result.stdout.decode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', dest="input_folder", help="Place where JSON files are stored.", type=str)
    parser.add_argument('--o', dest="output_text_file", help="File to store output results in text format.", type=str)
    parser.add_argument('--d', dest="output_dir", help="Output directory for output JSON files.", type=str)

    args = parser.parse_args()
    main(args.input_folder, args.output_text_file, args.output_dir)
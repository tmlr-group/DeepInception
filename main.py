import argparse
import json
import os

from conversers import load_attack_and_target_models

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "vicuna",
        help = "Name of target model.",
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type = int,
        default = 300,
        help = "Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--exp_name",
        type = str,
        default = "main",
        choices=['main', 'abl_c', 'abl_layer', 'multi_scene', 'abl_fig6_4'],
        help = "Experiment file name"
    )

    parser.add_argument(
        "--defense",
        type = str,
        default = "none",
        choices=['none', 'sr', 'ic'],
        help = "LLM defense: None, Self-Reminder, In-Context"
    )
    ##################################################
    args = parser.parse_args()

    
    f = open(f'./res/data{args.exp_name}.json',) 
    datas = json.load(f) 
    f.close() 
    results = [{} for _ in range(len(datas))]

    for idx, data in enumerate(datas):
        questions = [data['inception_attack']] + data['questions']
        targetLM = load_attack_and_target_models(args)
        results[idx]['topic'] = data['topic']
        # Get target responses
        results[idx]['qA_pairs'] = []
        for question in questions:
            target_response_list = targetLM.get_response(question, args.defense)
            results[idx]['qA_pairs'].append({'Q': question, 'A': target_response_list})

        del targetLM
    
    results_dumped = json.dumps(results)
    os.makedirs('results', exist_ok=True)
    with open(f'./results/{args.target_model}_{args.exp_name}_{args.defense}_results.json', 'w+') as f:
        f.write(results_dumped)
    f.close()

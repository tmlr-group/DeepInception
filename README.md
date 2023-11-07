# Deep Inception: Hypnotize LLM to Be Jailbreaker

[![arXiv](https://img.shields.io/badge/cs.ML-arXiv%3A2311.03191-b31b1b)](https://arxiv.org/abs/2311.03191)
</div>

## Abstract
Despite remarkable success in various applications, large language models (LLMs) are vulnerable to adversarial jailbreaks that make the safety guardrails void. However, previous studies for jailbreaks usually resort to brute-force optimization or extrapolations of a high computation cost, which might not be practical or effective. In this paper, inspired by the Milgram experiment that individuals can harm another person if they are told to do so by an authoritative figure, we disclose a lightweight method, termed as DeepInception, which can easily hypnotize LLM to be a jailbreaker and unlock its misusing risks. Specifically, DeepInception leverages the personification ability of LLM to construct a novel nested scene to behave, which realizes an adaptive way to escape the usage control in a normal scenario and provides the possibility for further direct jailbreaks. Empirically, we conduct comprehensive experiments to show its efficacy. Our DeepInception can achieve competitive jailbreak success rates with previous counterparts and realize a continuous jailbreak in subsequent interactions, which reveals the critical weakness of self-losing on both open/closed-source LLMs like Falcon, Vicuna, Llama-2, and GPT-3.5/4/4V. Our investigation appeals that people should pay more attention to the safety aspects of LLMs and a stronger defense against their misuse risks.

## Getting Started
To set up the enviroment, run

```
conda create -n deepinception python=3.10
conda activate deepinception
pip install -r requirements.txt
```

Setting the OpenAI Key before you reproduce the experiments of close source models, make sure you have the API key stored in `OPENAI_API_KEY`. For example,
```
export OPENAI_API_KEY=[YOUR_API_KEY_HERE]
```

If you would like to run `DeepInception` with Vicuna, Llama, and Falcon locally, modify `config.py` with the proper path of these three models. 

Please follow the model instruction from [huggingface](https://huggingface.co/) to download the models, inlcuding [Vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k), [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct).


## Run experiments
To run `DeepInception`, run
```
python3 main.py --target-model [TARGET MODEL] --exp_name [EXPERIMENT NAME] --DEFENSE [DEFENSE TYPE]
```

For example, to run main `DeepInception` experiments (Tab.1) with `Vicuna-v1.5-7b` as the target model with the default maximum number of tokens, run
```
python3 main.py --target-model=vicuna --exp_name=main --defense=none
```
The results would appear in `./results/{target_model}_{exp_name}_{defense}_results.json`, in this example is `./results/vicuna_main_none_results.json`

See `main.py` for all of the arguments and descriptions.

## Citation
```
@article{
    li2023deepinception,
    title={DeepInception: Hypnotize Large Language Model to Be Jailbreaker}, 
    author={Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, Bo Han},
    journal={arXiv preprint arXiv:2311.03191},
    year={2023}
}

```

## Acknowledgement

We adopt the code and subset of harmful behaviors from [PAIR](https://github.com/patrickrchao/JailbreakingLLMs).
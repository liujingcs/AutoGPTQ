import json
import time
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from typing import Dict, List, Optional

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor
from datasets import Dataset
import datautils
import math

logger = logging.getLogger(__name__)

random.seed(0)

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class CustomizedMinNewTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        min_new_tokens: int = None,
        eos_token_id: int = None,
    ):
        self.eos_token_id = eos_token_id
        self.min_new_tokens = min_new_tokens or 0
        self.current_step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.current_step += 1

        if self._skip_process():
            return scores

        if any(each is not None for each in [self.eos_token_id]):
            banned_mask = torch.zeros_like(scores).to(scores.device)
            if self.eos_token_id and self.current_step <= self.min_new_tokens:
                banned_mask = self._fill_banned_mask(input_ids, banned_mask, {1: [[self.eos_token_id]]})
            scores = scores.masked_fill(banned_mask.bool(), -float("inf"))

        return scores

    def _skip_process(self):
        if self.current_step > self.min_new_tokens:
            return True
        return False

    @staticmethod
    def _fill_banned_mask(
        input_ids: torch.LongTensor,
        banned_mask: torch.Tensor,
        len2words_ids: Dict[int, List[List[int]]]
    ):
        for token_len, token_ids in len2words_ids.items():
            if token_len == 1:
                banned_mask[..., list(chain(*token_ids))] = 1
            elif input_ids.shape[-1] < token_len - 1:
                continue
            else:
                token_ids = torch.LongTensor(token_ids).to(input_ids.device)
                hit_masks = torch.all(
                    token_ids[..., :-1].unsqueeze(0).repeat(input_ids.shape[0], 1, 1)
                    == input_ids[..., -(token_ids.shape[-1] - 1):].unsqueeze(1),
                    dim=-1
                )
                for idx in range(hit_masks.shape[0]):
                    selected_token_ids = torch.masked_select(token_ids[..., -1], hit_masks[idx])
                    if len(selected_token_ids):
                        banned_mask[idx, selected_token_ids] = 1
        return banned_mask


def load_data(data_path, tokenizer, n_samples, max_new_tokens):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
            "text": texts
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def load_model_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    from_pretrained: bool = False,
    max_memory: Optional[dict] = None,
    model_basename: Optional[str] = None,
    quantize_config: Optional[str] = None,
    trust_remote_code: bool = False,
    use_triton: bool = False,
    use_safetensors: bool = True,
    use_fast_tokenizer: bool = False,
    inject_fused_attention: bool = True,
    inject_fused_mlp: bool = True,
    disable_exllama: bool = False
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path or model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if from_pretrained:
        model = AutoGPTQForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            quantize_config=BaseQuantizeConfig(),
            max_memory=max_memory,
            trust_remote_code=trust_remote_code
        )
        model = model.to("cuda")
    else:
        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            use_triton=use_triton,
            inject_fused_attention=inject_fused_attention,
            inject_fused_mlp=inject_fused_mlp,
            use_cuda_fp16=True,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=False,
            disable_exllama=disable_exllama
        )

    return model, tokenizer


def llama_benchmark(model, testenc, check=False):
    model.config.use_cache = True
    input_ids = testenc.input_ids
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    seq_len = model.seqlen
    nsamples = input_ids.numel() // seq_len
    max_samples = 128
    nsamples = min(nsamples, max_samples)

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = torch.nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        seq_len = model.seqlen
        times = []
        tknps = []
        torch.cuda.cudart().cudaProfilerStart()

        for i in tqdm(range(nsamples), desc='Benchmarking', ncols=80):
            batch = input_ids[:, (i * seq_len):((i + 1) * seq_len)].to(DEV)
            torch.cuda.set_device(DEV)
            start_time = time.perf_counter()
            out = model(
                batch,
                past_key_values=None,
                attention_mask=attention_mask[:, :seq_len].reshape((1, -1))
            #     past_key_values=cache['past'],
            #     attention_mask=attention_mask[:, :(i + 1)*model.seqlen].reshape((1, -1))
            )
            sync()
            times.append(time.perf_counter() - start_time)
            tknps.append(batch.shape[-1] // times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        torch.cuda.cudart().cudaProfilerStop()
        import numpy as np
        print(f'Median times: {np.median(times)} +- {1.96 * np.std(times[2:-2])}')
        print(f'Median tokens/second: {np.median(tknps)} +- {1.96 * np.std(tknps[2:-2])}', )
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
        return np.median(times)


def llama_multigpu(model, gpus):
    model.model.model.embed_tokens = model.model.model.embed_tokens.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])
    if hasattr(model.model.model, 'norm') and model.model.model.norm is not None:
        model.model.model.norm = model.model.model.norm.to(gpus[-1])

    cache = {'mask': None, 'positions': None}

    class MoveModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                torch.cuda.set_device(self.dev)
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['positions'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
                cache['positions'] = kwargs['position_ids'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            kwargs['position_ids'] = cache['positions']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        print("Move layer {} to gpu: {}".format(i, gpus[i // pergpu]))
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))
        print(layers[i].device)

    model.gpus = gpus


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--from_pretrained", action="store_true")
    parser.add_argument("--model_basename", type=str, default=None)
    parser.add_argument("--quantize_config_save_dir", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--disable_exllama", action="store_true")
    parser.add_argument("--no_inject_fused_attention", action="store_true")
    parser.add_argument("--no_inject_fused_mlp", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--per_gpu_max_memory", type=int, default=None)
    parser.add_argument("--cpu_max_memory", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length used in evaluations and benchmarks.')
    args = parser.parse_args()

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    logger.info(f"max_memory: {max_memory}")

    quantize_config = None
    if args.quantize_config_save_dir:
        quantize_config = BaseQuantizeConfig.from_pretrained(args.quantize_config_save_dir)

    if args.use_safetensors:
        logger.warning("The command --use_safetensors is deprecated and will be removed in the next release. It is now by default activated.")

    logger.info("loading model and tokenizer")
    start = time.time()
    model, tokenizer = load_model_tokenizer(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        from_pretrained=args.from_pretrained,
        max_memory=max_memory,
        model_basename=args.model_basename,
        quantize_config=quantize_config,
        trust_remote_code=args.trust_remote_code,
        use_triton=args.use_triton,
        use_safetensors=True,
        use_fast_tokenizer=args.use_fast_tokenizer,
        inject_fused_attention=not args.no_inject_fused_attention,
        inject_fused_mlp=not args.no_inject_fused_mlp,
        disable_exllama=args.disable_exllama
    )
    end = time.time()
    logger.info(f"model and tokenizer loading time: {end - start:.4f}s")
    logger.info(f"model quantized: {model.quantized}")
    logger.info(f"quantize config: {model.quantize_config.to_dict()}")
    logger.info(f"model device map: {model.hf_device_map}")

    if args.use_triton:
        logger.info("warmup triton, this may take a while.")
        model.warmup_triton()

    dataset = 'wikitext2'
    dataloader, testloader = datautils.get_loaders(
        dataset, seed=args.seed, model=args.model_name_or_path, seqlen=model.seqlen,
    )
    gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
    if len(gpus) > 1:
        llama_multigpu(model, gpus)
    else:
        model = model.to(DEV)
    llama_benchmark(model, testloader, check=False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

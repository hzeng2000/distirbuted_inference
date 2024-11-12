"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple
import evaluate
from typing import Any, Dict, List
from transformers import AutoTokenizer

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser
from vllm.inputs import PromptInputs
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataset_sampler import sample_OpenOrca_requests
import yaml
import ray 
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
import numpy as np
from functools import partial
ray.init()
num_instances = 1

# Function to expand environment variables in a config dictionary
def expand_env_vars(config):
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)  # This will replace ${HOME} with the actual home directory
    else:
        return config

class LLMPredictor:

    def __init__(self, 
                model: str,
                tokenizer: str,
                quantization: Optional[str],
                tensor_parallel_size: int,
                seed: int,
                n: int,
                use_beam_search: bool,
                trust_remote_code: bool,
                dtype: str,
                max_model_len: Optional[int],
                enforce_eager: bool,
                kv_cache_dtype: str,
                quantization_param_path: Optional[str],
                device: str,
                enable_prefix_caching: bool,
                enable_chunked_prefill: bool,
                max_num_batched_tokens: int,
                distributed_executor_backend: Optional[str],
                gpu_memory_utilization: float = 0.9,
                num_scheduler_steps: int = 1,
                use_v2_block_manager: bool = False,
                download_dir: Optional[str] = None,
                load_format: str = EngineArgs.load_format,
                disable_async_output_proc: bool = False,
                warmup_iter: int = 10,
        ):
        # Create an LLM.
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.n = n
        self.use_beam_search = use_beam_search
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer,
            quantization=quantization,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            quantization_param_path=quantization_param_path,
            device=device,
            enable_prefix_caching=enable_prefix_caching,
            download_dir=download_dir,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            distributed_executor_backend=distributed_executor_backend,
            load_format=load_format,
            num_scheduler_steps=num_scheduler_steps,
            use_v2_block_manager=use_v2_block_manager,
            disable_async_output_proc=disable_async_output_proc,
        )
        self.warmup_iter = warmup_iter

    def __call__(self, requests: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        sampling_params: List[SamplingParams] = []
        references: List[str] = []
        prompts: List[str] = []
        generated_text: List[str] = []
        output_length: List[int] = []
        item = requests["item"].tolist()   
        for prompt, _, output_len, reference in item:
            # print(f"Prompt: {prompt}, output_len: {output_len}, reference: {reference}")
            prompts.append(prompt)
            references.append(reference)
            output_len = int(output_len)  
            sampling_params.append(
                SamplingParams(
                    n=self.n,
                    temperature=0.0 if self.use_beam_search else 1.0,
                    top_p=1.0,
                    use_beam_search=self.use_beam_search,
                    ignore_eos=True,
                    min_tokens=int(output_len * 0.9),
                    max_tokens=output_len,
                )
            )
        print("Warmup ......")
        self.llm.generate(prompts[:self.warmup_iter], sampling_params[:self.warmup_iter], use_tqdm=True)
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)
        end = time.perf_counter()
        elapsed_time = []
        for output in outputs:
            # prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
            output_length.append(len(output.outputs[0].token_ids))
            elapsed_time.append(end - start)
        return {
            "elapsed_time": elapsed_time,
            "prompts": prompts,
            "generated_text": generated_text,
            "references": references,
            "output_length": output_length
        }

    
def run_vllm(
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    quantization_param_path: Optional[str],
    device: str,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    distributed_executor_backend: Optional[str],
    gpu_memory_utilization: float = 0.9,
    num_scheduler_steps: int = 1,
    use_v2_block_manager: bool = False,
    download_dir: Optional[str] = None,
    load_format: str = EngineArgs.load_format,
    disable_async_output_proc: bool = False,
    dataset: Optional[str] = None,
    dataset_path: str = "",
    num_prompts: int = 10,
    input_len: int = 10,
    output_len: int = 10,
    warmup_iter: int = 10,
) -> Tuple[float, List[Tuple[str, int, int, str]]]:
    from vllm import LLM, SamplingParams
    tokenizer_str = tokenizer
    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=args.trust_remote_code)
    if dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (input_len - 1)
        reference = "hello" * output_len
        requests = [(prompt, input_len, output_len, reference)
                    for _ in range(num_prompts)]
    else:
        if dataset == "OpenOrca":
            requests = sample_OpenOrca_requests(dataset_path, num_prompts, tokenizer, output_len)
            warmup_requests = sample_OpenOrca_requests(dataset_path, warmup_iter, tokenizer, output_len)
        else:
            raise ValueError("Unsupported dataset. Please choose 'OpenOrca'.")
    
    # for prompt, _, output_len, reference in requests:
    #     with open("reference_before.txt", "a") as file:
    #         file.write(f"reference_before: {reference}\n")
    # distributed generation benchmark thoughput
    requests = requests * num_instances
    ds = ray.data.from_items(requests)
    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        fn_constructor_args=[
            model,
            tokenizer_str,
            quantization,
            tensor_parallel_size,
            seed,
            n,
            use_beam_search,
            trust_remote_code,
            dtype,
            max_model_len,
            enforce_eager,
            kv_cache_dtype,
            quantization_param_path,
            device,
            enable_prefix_caching,
            enable_chunked_prefill,
            max_num_batched_tokens,
            distributed_executor_backend,
            gpu_memory_utilization,
            num_scheduler_steps,
            use_v2_block_manager,
            download_dir,
            load_format,
            disable_async_output_proc,
            warmup_iter,
            ], 
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=len(requests) / num_instances,
        **resources_kwarg,
    )
    all_outputs = ds.to_pandas()
    # print(f"all_outputs: {all_outputs}")
    # extract outputs
    elapsed_times = all_outputs["elapsed_time"]
    tho_elapsed = max(elapsed_times)
    # with open("dis_tho_elapsed.txt", "a") as file:
    #     file.write(f"dis_tho_elapsed: {tho_elapsed}\n")
            
    prompts = all_outputs["prompts"]
    generated_texts = all_outputs["generated_text"].to_list()
    references = all_outputs["references"].to_list()
    # with open("references.txt", "a") as file:
    #     for reference in references:
    #         file.write(f"reference: {reference}\n")
    output_length = all_outputs["output_length"].to_list()
    # print(output_length)
    
    # # latency benchmark
    llm = LLM(
        model=model,
        tokenizer=tokenizer_str,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        quantization_param_path=quantization_param_path,
        device=device,
        enable_prefix_caching=enable_prefix_caching,
        download_dir=download_dir,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        distributed_executor_backend=distributed_executor_backend,
        load_format=load_format,
        num_scheduler_steps=num_scheduler_steps,
        use_v2_block_manager=use_v2_block_manager,
        disable_async_output_proc=disable_async_output_proc,
    )
    start = time.perf_counter()
    half = int(len(requests)/num_instances/2)
    for prompt, _, output_len, reference in requests[:half]:
        sampling_param=SamplingParams(
                n=n,
                temperature=0.0 if use_beam_search else 1.0,
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=True,
                min_tokens=output_len*0.9,
                max_tokens=output_len,
            )
        llm.generate(prompt, sampling_param, use_tqdm=False)
    end = time.perf_counter()
    latency_elapsed = end - start
    
    # ROUGE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(script_dir, "..", "utils")
    rouge = evaluate.load(os.path.join(utils_dir, "rouge"))
    rouge_results = rouge.compute(predictions=generated_texts, references=references)

    for i in range(len(output_length)):
        requests[i] = (requests[i][0], requests[i][1], output_length[i], requests[i][3])

    return latency_elapsed, tho_elapsed, requests, rouge_results
    # return 0, tho_elapsed, requests, rouge_results


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    if args.backend == "vllm":
        latency_elapsed, tho_elapsed, requests, rouge_results = run_vllm(
            args.model, args.tokenizer, args.quantization,
            args.tensor_parallel_size, args.seed, args.n, args.use_beam_search,
            args.trust_remote_code, args.dtype, args.max_model_len,
            args.enforce_eager, args.kv_cache_dtype,
            args.quantization_param_path, args.device,
            args.enable_prefix_caching, args.enable_chunked_prefill,
            args.max_num_batched_tokens, args.distributed_executor_backend,
            args.gpu_memory_utilization, args.num_scheduler_steps,
            args.use_v2_block_manager, args.download_dir, args.load_format,
            args.disable_async_output_proc, args.dataset, 
            args.dataset_path, args.num_prompts, args.input_len, args.output_len,
            args.warmup_iter)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_prompt_tokens = sum(prompt_len
                           for _, prompt_len, output_len, _ in requests)
    total_output_tokens = sum(output_len
                           for _, prompt_len, output_len, _ in requests)
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len, _ in requests)
    # with open("dis_total_num_tokens.txt", "a") as file:
    #     file.write(f"dis_total_num_tokens: {total_num_tokens}\n")

    print(f"Throughput: {len(requests) / tho_elapsed:.2f} requests/s "
        f"({total_num_tokens / tho_elapsed:.2f} tokens/s) | "
        f"latency: {latency_elapsed / ((len(requests)/num_instances/2)):.2f} s/request "
        f"ROUGE-1: {rouge_results['rouge1']:.2f}, "
        f"ROUGE-2: {rouge_results['rouge2']:.2f}, "
        f"ROUGE-L: {rouge_results['rougeL']:.2f}")

    
    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": tho_elapsed,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / tho_elapsed,
            "tokens_per_second": total_num_tokens / tho_elapsed,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default="OpenOrca",
                        choices=["OpenOrca"],
                        help="dataset name.")
    parser.add_argument("--dataset-path",
                        type=str,
                        default="~/AIPerf_Inference_Datasets/OpenOrca/1M-GPT4-Augmented.parquet",
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", 
                        type=str, 
                        default="~/AIPerf-Inference_Models/Llama-2-7b-hf")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=128,
                        help="Number of prompts to process.")
    parser.add_argument("--warmup_iter",
                        type=int,
                        default=10,
                        help="Number of prompts to warmup.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "openvino", "tpu", "xpu"],
        help='device type for vLLM execution, supporting CUDA, OpenVINO and '
        'CPU.')
    parser.add_argument(
        "--num-scheduler-steps",
        type=int,
        default=1,
        help="Maximum number of forward steps per scheduler call.")
    parser.add_argument("--use-v2-block-manager",
                        action='store_true',
                        help="Enable block manager v2.")
    parser.add_argument(
        "--enable-prefix-caching",
        action='store_true',
        help="Enable automatic prefix caching for vLLM backend.")
    parser.add_argument("--enable-chunked-prefill",
                        action='store_true',
                        # default=False,
                        help="enable chunked prefill for vLLM backend.")
    parser.add_argument('--max-num-batched-tokens',
                        type=int,
                        default=None,
                        help='maximum number of batched tokens per '
                        'iteration')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument(
        '--load-format',
        type=str,
        default=EngineArgs.load_format,
        choices=[
            'auto', 'pt', 'safetensors', 'npcache', 'dummy', 'tensorizer',
            'bitsandbytes'
        ],
        help='The format of the model weights to load.\n\n'
        '* "auto" will try to load the weights in the safetensors format '
        'and fall back to the pytorch bin format if safetensors format '
        'is not available.\n'
        '* "pt" will load the weights in the pytorch bin format.\n'
        '* "safetensors" will load the weights in the safetensors format.\n'
        '* "npcache" will load the weights in pytorch format and store '
        'a numpy cache to speed up the loading.\n'
        '* "dummy" will initialize the weights with random values, '
        'which is mainly for profiling.\n'
        '* "tensorizer" will load the weights using tensorizer from '
        'CoreWeave. See the Tensorize vLLM Model script in the Examples'
        'section for more information.\n'
        '* "bitsandbytes" will load the weights using bitsandbytes '
        'quantization.\n')
    parser.add_argument(
        "--disable-async-output-proc",
        action='store_true',
        default=False,
        help="Disable async output processor for vLLM backend.")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = os.path.join(script_dir, "config.yaml")
    with open(config, "r") as f:
        config = expand_env_vars(yaml.safe_load(f))
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None
    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 1
            }] * args.tensor_parallel_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))


    resources_kwarg: Dict[str, Any] = {}
    if args.tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn
    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)

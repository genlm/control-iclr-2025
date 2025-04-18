import json
import click
import asyncio
import warnings
from typing import Type
from genlm.eval.core import Dataset, Evaluator, run_evaluation
from syn_sem_control.util import mean_ci_results


def common_options(f):
    """Shared CLI options across all domains"""
    f = click.option(
        "--model-type",
        required=True,
        help="Type of model to use (base, lcd, grammar-only-is, grammar-only-smc, sample-rerank, full-is, full-smc).",
    )(f)
    f = click.option(
        "--lm-name",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the language model to use.",
    )(f)
    f = click.option(
        "--max-tokens",
        default=100,
        help="Maximum number of tokens to generate.",
    )(f)
    f = click.option(
        "--output-dir",
        default=None,
        help="Directory to write the inference results.",
    )(f)
    f = click.option(
        "--overwrite-results",
        is_flag=True,
        help="Overwrite existing evaluation results.",
    )(f)
    f = click.option(
        "--overwrite-outputs",
        is_flag=True,
        help="Overwrite existing inference output.",
    )(f)
    f = click.option(
        "--lm-args",
        default='{}',
        help="Arguments for genlm.control.PromptedLLM initialization, in json format.",
    )(f)
    f = click.option(
        "--verbosity",
        default=1,
        type=int,
        help="Verbosity level for evaluation.",
    )(f)
    f = click.option(
        "--sampler-cache-size",
        default=1,
        type=int,
        help="Size of LRU cache for initialized samplers.",
    )(f)
    f = click.option(
        "--critic-cache-size",
        default=1,
        type=int,
        help="Size of LRU cache for initialized critics.",
    )(f)
    f = click.option(
        "--n-replicates",
        default=1,
        type=int,
        help="Number of runs of each instance for evaluation.",
    )(f)
    f = click.option(
        "--n-particles",
        default=1,
        type=int,
        help="Number of particles (for IS/SMC models).",
    )(f)
    f = click.option(
        "--ess-threshold",
        default=0.0,
        type=float,
        help="ESS threshold (for SMC models).",
    )(f)
    f = click.option(
        "--resampling-method",
        default=None,
        type=str,
        help="Resampling method (for SMC models).",
    )(f)
    return f


def setup_model_and_params(model_type: str, kwargs: dict, model_classes: dict):
    """Helper function to set up model class and adjust sampling parameters."""
    model_class, needs_particles, needs_ess, needs_resampling_method = model_classes[model_type]
    
    # Adjust sampling parameters if not needed, warning if they were set
    if not needs_particles and kwargs.get("n_particles", 1) != 1:
        warnings.warn(f"Model type '{model_type}' doesn't use particles. Overriding n_particles=1")
        kwargs["n_particles"] = 1
        
    if not needs_ess and kwargs.get("ess_threshold", 0.0) != 0.0:
        warnings.warn(f"Model type '{model_type}' doesn't use ESS. Overriding ess_threshold=0.0")
        kwargs["ess_threshold"] = 0.0
        
    if not needs_resampling_method and kwargs.get("resampling_method", None) is not None:
        warnings.warn(f"Model type '{model_type}' doesn't use resampling. Provided resampling method will be ignored.")
        kwargs["resampling_method"] = None
    
    return model_class, kwargs


def run_model_evaluation(
    dataset: Dataset,
    model_class: Type,
    evaluator: Evaluator,
    lm_name: str,
    max_tokens: int,
    n_particles: int,
    ess_threshold: float,
    resampling_method: str,
    n_replicates: int,
    output_dir: str,
    overwrite_results: bool,
    overwrite_outputs: bool,
    sampler_cache_size: int,
    critic_cache_size: int,
    lm_args: str,
    verbosity: int,
):
    """Common evaluation logic across domains"""
    model = model_class(
        model_name=lm_name,
        ess_threshold=ess_threshold,
        n_particles=n_particles,
        max_tokens=max_tokens,
        lm_args=json.loads(lm_args),
        sampler_cache_size=sampler_cache_size,
        critic_cache_size=critic_cache_size,
        # default to multinomial resampling method, but this will not be used for models with ess_threshold=0.0 or n_particles=1
        resampling_method=resampling_method if resampling_method is not None else 'multinomial',
    )

    results = asyncio.run(run_evaluation(
        dataset=dataset,
        model=model,
        evaluator=evaluator,
        n_replicates=n_replicates,
        verbosity=verbosity,
        overwrite_results=overwrite_results,
        overwrite_outputs=overwrite_outputs,
        output_dir=output_dir,
    ))

    mean, lower, upper = mean_ci_results(results)
    print(f"Mean weighted accuracy: {mean}")
    print(f"95% CI: ({lower}, {upper})")
    return
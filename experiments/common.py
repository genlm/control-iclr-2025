import json
import click
import asyncio
import warnings
from typing import Type, Callable
from genlm.eval.core import Dataset, Evaluator, run_evaluation

from .util import mean_ci_results
from . import models


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
        default="{}",
        help="Arguments for genlm.control.PromptedLLM initialization, in json format.",
    )(f)
    f = click.option(
        "--verbosity",
        default=1,
        type=int,
        help="Verbosity level for evaluation.",
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
    f = click.option(
        "--max-instances",
        default=100000,
        type=int,
        help="Maximum number of instances in the dataset to evaluate.",
    )(f)
    return f


MODEL_CLASSES = {
    # model_type : (class, needs_particles, needs_ess, needs_resampling_method)
    "base": (models.BaseLM, False, False, False),
    "lcd": (models.FastImproperlyWeighted, False, False, False),
    "grammar-only-is": (models.FastProperlyWeighted, True, False, False),
    "grammar-only-smc": (models.FastProperlyWeighted, True, True, True),
    "sample-rerank": (models.FullImproperlyWeighted, True, False, False),
    "full-is": (models.FullProperlyWeighted, True, False, False),
    "full-smc": (models.FullProperlyWeighted, True, True, True),
    "critic-is":  (models.DirectProperlyWeighted, True,  False, False),
    "critic-smc": (models.DirectProperlyWeighted, True,  True,  True),
}


def setup_model_and_params(model_type: str, kwargs: dict):
    """Helper function to set up model class and adjust sampling parameters.

    Args:
        model_type (str): Type of model to use
        kwargs (dict): Model parameters
        model_classes (dict): Dictionary mapping model types to (class, needs_particles, needs_ess, needs_resampling)

    Returns:
        tuple: (model_class, validated_kwargs)
    """
    model_class, needs_particles, needs_ess, needs_resampling = MODEL_CLASSES[
        model_type
    ]

    # Validate and adjust parameters based on model requirements
    if not needs_particles:
        if kwargs.get("n_particles", 1) != 1:
            warnings.warn(
                f"Model type '{model_type}' doesn't use particles. Setting n_particles=1"
            )
        kwargs["n_particles"] = 1

    if not needs_ess:
        if kwargs.get("ess_threshold", 0.0) != 0.0:
            warnings.warn(
                f"Model type '{model_type}' doesn't use ESS. Setting ess_threshold=0.0"
            )
        kwargs["ess_threshold"] = 0.0

    if not needs_resampling:
        if kwargs.get("resampling_method") is not None:
            warnings.warn(
                f"Model type '{model_type}' doesn't use resampling. Setting resampling_method=None"
            )
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
    lm_args: str,
    verbosity: int,
    potential_factory: models.PotentialFactory,
    cache_key_fn: Callable,
    prompt_formatter: Callable,
    max_instances: int = float("inf"),
    max_cache_size: int = 1,
    **kwargs,
):
    """Common evaluation logic across domains"""
    # default to multinomial resampling method, but this will not be used for models with ess_threshold=0.0 or n_particles=1
    resampling_method = resampling_method or "multinomial"

    model = model_class(
        # Language model parameters
        lm_name=lm_name,
        lm_args=json.loads(lm_args),
        # Sampling parameters
        ess_threshold=ess_threshold,
        n_particles=n_particles,
        max_tokens=max_tokens,
        resampling_method=resampling_method,
        # Potential factory
        potential_factory=potential_factory,
        # Prompt formatter
        prompt_formatter=prompt_formatter,
        # Caching parameters
        cache_key_fn=cache_key_fn,
        max_cache_size=max_cache_size,
        # Other parameters
        **kwargs,
    )

    results = asyncio.run(
        run_evaluation(
            dataset=dataset,
            model=model,
            evaluator=evaluator,
            n_replicates=n_replicates,
            verbosity=verbosity,
            max_instances=max_instances,
            overwrite_results=overwrite_results,
            overwrite_outputs=overwrite_outputs,
            output_dir=output_dir,
        )
    )

    mean, lower, upper = mean_ci_results(results)
    print(f"Mean weighted accuracy: {mean}")
    print(f"95% CI: ({lower}, {upper})")
    return

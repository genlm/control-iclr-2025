import click
from genlm.eval.domains.ds1000 import (
    DS1000Dataset,
    DS1000Evaluator, 
    DS1000RuntimeNoErrorPotential, 
    TrivialPotential,
    default_prompt_formatter
)

from experiments.models import PotentialFactory
from experiments.common import (
    common_options,
    run_model_evaluation,
    setup_model_and_params,
)
from experiments.util import make_prompt_formatter


class DS1000PotentialFactory(PotentialFactory):
    def __init__(self, timeout_s=5):
        self.timeout_s = timeout_s

    def get_fast_potential(self, instance):
        return TrivialPotential()

    def get_expensive_potential(self, instance):
        return DS1000RuntimeNoErrorPotential(
            code_context=instance.code_context,
            timeout_seconds=self.timeout_s,
        )


@click.command(help="Run DS1000 evaluation.")
@common_options
@click.option(
    "--lm-name",
    default="meta-llama/Meta-Llama-3.1-8B",
    help="Name of the language model to use.",
)
@click.option(
    "--max-tokens",
    default=1024,
    help="Maximum number of tokens to generate.",
)

def main(**kwargs):
    model_type = kwargs.pop("model_type")

    model_class, kwargs = setup_model_and_params(model_type, kwargs)
    dataset = DS1000Dataset.from_hf(
        split="test",
        shuffle=True,
        seed=1234,
    )
    evaluator = DS1000Evaluator()

    def cache_key_fn(instance):
        return "always_true"

    prompt_formatter = make_prompt_formatter(
        kwargs.get("lm_name", ""), default_prompt_formatter
    )

    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        potential_factory=DS1000PotentialFactory(),
        cache_key_fn=cache_key_fn,
        prompt_formatter=prompt_formatter,
        **kwargs,
    )


if __name__ == "__main__":
    main()

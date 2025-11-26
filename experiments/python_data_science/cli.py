import click
from pathlib import Path
import os

from genlm.eval.domains.ds1000 import (
    DS1000Dataset,
    DS1000Evaluator,
    DS1000RuntimeNoErrorPotential,
    default_prompt_formatter,
)

from experiments.models import PotentialFactory
from experiments.common import (
    common_options,
    run_model_evaluation,
    setup_model_and_params,
)
from experiments.util import make_prompt_formatter


class DS1000PotentialFactory(PotentialFactory):
    def __init__(self, env_py, timeout_s=15):
        self.env_py = str(env_py)
        self.timeout_s = timeout_s

    def get_fast_potential(self, instance):  # No trivial potential in DS1000
        pass

    def get_expensive_potential(self, instance):
        return DS1000RuntimeNoErrorPotential(
            code_context=instance.code_context,
            python_executable=self.env_py,
            extra_env={"PYTHONHASHSEED": "0"},
            timeout_seconds=self.timeout_s,
        )


@click.command(help="Run DS1000 evaluation.")
@click.option(
    "--libraries",
    default=None,
    multiple=True,
    help="Libraries to include in the environment, comma-separated (e.g. numpy,pandas).",
)
@common_options
@click.option(
    "--lm-name",
    default="meta-llama/Meta-Llama-3-8B",
    help="Name of the language model to use.",
)
@click.option(
    "--max-tokens",
    default=500,
    help="Maximum number of tokens to generate.",
)
def main(**kwargs):
    model_type = kwargs.pop("model_type")

    model_class, kwargs = setup_model_and_params(model_type, kwargs)
    libraries = kwargs.pop("libraries")
    dataset = DS1000Dataset.from_hf(
        libraries=libraries,
        split="test",
    )

    root = Path.cwd()
    env_py = (
        root
        / ".ds1000env"
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    print(env_py, env_py.exists())

    evaluator = DS1000Evaluator(
        python_executable=str(env_py),
        timeout_seconds=25.0,
        extra_env={"PYTHONHASHSEED": "0"},
    )

    def cache_key_fn(instance):
        return "always_true"

    prompt_formatter = make_prompt_formatter(
        kwargs.get("lm_name", ""), default_prompt_formatter
    )

    # EOS tokens
    def eos_token_factory(llm):
        return [
            t
            for t in llm.vocab
            if b"</code>" in t
            or b"</code>".startswith(t)
            or b"END SOLUTION" in t
            or b"END SOLUTION".startswith(t)
        ]

    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        potential_factory=DS1000PotentialFactory(env_py=env_py),
        cache_key_fn=cache_key_fn,
        eos_token_factory=eos_token_factory,  # Agressive EOS
        prompt_formatter=prompt_formatter,
        **kwargs,
    )


if __name__ == "__main__":
    main()

import click
from genlm.control import BoolCFG
from genlm.eval.domains.molecular_synthesis import (
    MolecularSynthesisDataset,
    PartialSmiles,
    MolecularSynthesisEvaluator,
    default_prompt_formatter,
)

from syn_sem_control.models import PotentialFactory
from syn_sem_control.common import (
    common_options,
    run_model_evaluation,
    setup_model_and_params,
)
from syn_sem_control.util import make_prompt_formatter


class MolecularSynthesisPotentialFactory(PotentialFactory):
    def __init__(self, grammar_path):
        with open(grammar_path, "r") as f:
            self.grammar = f.read()

    def get_fast_potential(self, instance):
        return BoolCFG.from_lark(self.grammar)

    def get_expensive_potential(self, instance):
        return PartialSmiles()


@click.command(help="Run Molecular Synthesis evaluation.")
@click.option(
    "--smiles-file",
    help="Path to the SMILES file.",
)
@common_options
@click.option(
    "--lm-name",
    default="meta-llama/Meta-Llama-3.1-8B",  # Override common default
    help="Name of the language model to use.",
)
@click.option(
    "--max-tokens",  # Override common default
    default=40,
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--grammar-path",  # Override common default
    default="smiles.lark",
    help="Path to the grammar file.",
)
def main(**kwargs):
    model_type = kwargs.pop("model_type")
    data_dir = kwargs.pop("smiles_file")

    model_class, kwargs = setup_model_and_params(model_type, kwargs)
    dataset = MolecularSynthesisDataset.from_smiles(data_dir)
    evaluator = MolecularSynthesisEvaluator()

    potential_factory = MolecularSynthesisPotentialFactory(kwargs.pop("grammar_path"))

    def cache_key_fn(instance):
        return "always_true"

    prompt_formatter = make_prompt_formatter(
        kwargs.get("lm_name", ""), default_prompt_formatter
    )

    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        potential_factory=potential_factory,
        cache_key_fn=cache_key_fn,
        prompt_formatter=prompt_formatter,
        **kwargs,
    )


if __name__ == "__main__":
    main()

import click
from genlm.eval.domains.molecular_synthesis.dataset import MolecularSynthesisDataset
from genlm.eval.domains.molecular_synthesis.evaluator import MolecularSynthesisEvaluator
from syn_sem_control.molecular_synthesis import models
from syn_sem_control.common import common_options, run_model_evaluation, setup_model_and_params

MODEL_CLASSES = {
    "base": (models.BaseLM, False, False, False),  # (class, needs_particles, needs_ess, needs_resampling_method)
    "lcd": (models.GrammarOnlyImproperlyWeighted, False, False, False),
    "grammar-only-is": (models.GrammarOnlyProperlyWeighted, True, False, False),
    "grammar-only-smc": (models.GrammarOnlyProperlyWeighted, True, True, True),
    "sample-rerank": (models.FullModelImproperlyWeighted, True, False, False),
    "full-is": (models.FullModel, True, False, False),
    "full-smc": (models.FullModel, True, True, True),
}

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
    "--max-tokens", # Override common default
    default=40,
    help="Maximum number of tokens to generate.",
)
def main(**kwargs):
    model_type = kwargs.pop("model_type")
    data_dir = kwargs.pop("smiles_file")
    
    model_class, kwargs = setup_model_and_params(model_type, kwargs, MODEL_CLASSES)    
    dataset = MolecularSynthesisDataset.from_smiles(data_dir)
    evaluator = MolecularSynthesisEvaluator()
    
    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        **kwargs
    )

if __name__ == "__main__":
    main()
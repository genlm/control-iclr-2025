import os
import click
from genlm.eval.domains.spider import SpiderDataset, SpiderEvaluator

from syn_sem_control.text_to_sql import models
from syn_sem_control.common import common_options, run_model_evaluation, setup_model_and_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_CLASSES = {
    # model_type : (class, needs_particles, needs_ess, needs_resampling_method)
    "base": (models.BaseLM, False, False, False),
    "lcd": (models.GrammarOnlyImproperlyWeighted, False, False, False),
    "grammar-only-is": (models.GrammarOnlyProperlyWeighted, True, False, False),
    "grammar-only-smc": (models.GrammarOnlyProperlyWeighted, True, True, True),
    "sample-rerank": (models.FullModelImproperlyWeighted, True, False, False),
    "full-is": (models.FullModel, True, False, False),
    "full-smc": (models.FullModel, True, True, True),
}

# Use `python cli.py --help` to see the available options.
@click.command(help="Run Spider (Text-to-SQL) evaluation.")
@click.option(
    "--spider-data-dir",
    default="spider_data",
    help="Path to the Spider dataset directory.",
)
@click.option(
    "--max-tokens",
    default=100,
    help="Maximum number of tokens to generate.",
)
@common_options
def main(**kwargs):
    model_type = kwargs.pop("model_type")
    data_dir = kwargs.pop("spider_data_dir")
    
    model_class, kwargs = setup_model_and_params(model_type, kwargs, MODEL_CLASSES)
    dataset = SpiderDataset.from_spider_dir(data_dir)
    evaluator = SpiderEvaluator(data_dir)
    
    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        **kwargs
    )

if __name__ == "__main__":
    main()
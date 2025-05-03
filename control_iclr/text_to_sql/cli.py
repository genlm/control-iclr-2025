import os
import click
from genlm.control import BoolCFG
from genlm.eval.domains.spider import (
    SpiderTableColumnVerifier,
    SpiderDataset,
    SpiderEvaluator,
    default_prompt_formatter,
)

from control_iclr.models import PotentialFactory
from control_iclr.common import (
    common_options,
    run_model_evaluation,
    setup_model_and_params,
)
from control_iclr.util import make_prompt_formatter

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SpiderPotentialFactory(PotentialFactory):
    def get_fast_potential(self, instance):
        assert instance.lark_grammar is not None
        return BoolCFG.from_lark(instance.lark_grammar)

    def get_expensive_potential(self, instance):
        assert instance.lark_grammar is not None
        return SpiderTableColumnVerifier(
            grammar=instance.lark_grammar, tables=instance.tables
        )


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
@click.option(
    "--spider-grammar-path",
    default="grammars.json",
    help="Path to the Spider grammar file.",
)
@common_options
def main(**kwargs):
    model_type = kwargs.pop("model_type")
    data_dir = kwargs.pop("spider_data_dir")
    grammar_path = kwargs.pop("spider_grammar_path")

    model_class, kwargs = setup_model_and_params(model_type, kwargs)
    dataset = SpiderDataset.from_spider_dir(data_dir, grammar_path)
    evaluator = SpiderEvaluator(data_dir)

    def cache_key_fn(instance):
        return instance.schema_name

    prompt_formatter = make_prompt_formatter(
        kwargs.get("lm_name", ""), default_prompt_formatter
    )

    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        potential_factory=SpiderPotentialFactory(),
        cache_key_fn=cache_key_fn,
        prompt_formatter=prompt_formatter,
        **kwargs,
    )


if __name__ == "__main__":
    main()

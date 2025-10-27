import click
from pathlib import Path
from genlm.control import BoolCFG

from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceEvaluator,
    GoalInferenceVALPotential,
    goal_default_prompt_formatter,
)

from experiments.models import PotentialFactory
from experiments.common import (
    common_options,
    run_model_evaluation,
    setup_model_and_params,
)
from experiments.util import make_prompt_formatter


class GoalInferencePotentialFactory(PotentialFactory):
    """
    Produces the fast (grammar/static) and expensive (plan validation) potentials.
    If the instance provides a lark_grammar, that is preferred; otherwise use grammar_text.
    If no plan is available or expensive checks are disabled, returns a NeutralPotential.
    """

    def __init__(
        self,
        domain_path: str,
        goal_grammar_text: str,
        fast_downward_cmd: str = "./fast-downward.sif",
        val_cmd: str = "Validate",
        cache_root: str | Path = "./cache",
        verbosity: int = 0,
        timeout_seconds: float = 180.0,
    ):
        self.goal_grammar_text = goal_grammar_text
        self.val_cmd = val_cmd
        self.fast_downward_cmd = fast_downward_cmd
        self.cache_root = cache_root
        self.verbosity = verbosity
        self.timeout_seconds = timeout_seconds

        with open(domain_path) as f:
            self.domain_text = f.read()

    def get_fast_potential(self, instance):
        return BoolCFG.from_lark(self.goal_grammar_text)

    def get_expensive_potential(self, instance):
        return GoalInferenceVALPotential(
            domain_pddl_text=self.domain_text,
            problem_pddl_text=instance.problem_text,
            fast_downward_cmd=self.fast_downward_cmd,
            val_cmd=self.val_cmd,
            cache_root=self.cache_root,
            verbosity=self.verbosity,
        )


@click.command(help="Run Goal Inference evaluation.")
@common_options
@click.option(
    "--lm-name",
    default="meta-llama/Meta-Llama-3.1-8B",
    help="Name of the language model to use.",
)
@click.option(
    "--grammar-path",
    default="grammars/goal_inference.lark",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional path to a Lark grammar file for goals (used as fallback).",
)
@click.option(
    "--max-tokens",
    default=150,
    help="Maximum number of tokens to generate.",
)
def main(**kwargs):
    # Pull common & task-specific args
    model_type = kwargs.pop("model_type")
    grammar_path = kwargs.pop("grammar_path", None)
    verbosity = int(kwargs.pop("verbosity", 0))

    max_objects = 9
    domains_opt = ["blocksworld"]

    # Load optional grammar text
    grammar_text = None
    if grammar_path is not None:
        grammar_text = Path(grammar_path).read_text(encoding="utf-8")

    # Setup model
    model_class, kwargs = setup_model_and_params(model_type, kwargs)

    dataset = GoalInferenceDataset.from_hf_planetarium(
        max_objects=max_objects,
        domains=domains_opt,
    )

    # Evaluator
    evaluator = GoalInferenceEvaluator()

    # Potentials
    domain_path = "pddl_domains/blocksworld.pddl"
    potential_factory = GoalInferencePotentialFactory(
        domain_path=domain_path,
        goal_grammar_text=grammar_text,
        fast_downward_cmd="./fast-downward.sif",
    )

    def cache_key_fn(instance):
        stem = Path(getattr(instance, "problem_path", "prob")).stem
        return f"{getattr(instance, 'instance_id', 'na')}::{stem}"

    # Prompt formatting
    prompt_formatter = make_prompt_formatter(
        kwargs.get("lm_name", ""),
        goal_default_prompt_formatter,
    )

    # EOS tokens (stop on newline by default)
    def eos_token_factory(llm):
        return [t for t in llm.vocab if b"))" in t]

    run_model_evaluation(
        dataset=dataset,
        model_class=model_class,
        evaluator=evaluator,
        potential_factory=potential_factory,
        cache_key_fn=cache_key_fn,
        prompt_formatter=prompt_formatter,
        eos_token_factory=eos_token_factory,
        verbosity=verbosity,
        **kwargs,
    )


if __name__ == "__main__":
    main()

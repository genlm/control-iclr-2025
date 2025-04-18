import numpy as np
from genlm.eval.util import bootstrap_ci
from genlm.control.sampler import EagerSetSampler, SetTokenSampler


class ImproperlyWeightedSetTokenSampler(SetTokenSampler):
    async def sample(self, context):
        x, _, logp = await super().sample(context)
        return x, 0, logp
    

def improperly_weighted_eager_token_sampler(llm, bool_cfg):
    return ImproperlyWeightedSetTokenSampler(EagerSetSampler(llm, bool_cfg))


def mean_ci_results(results, ci=0.95, n_bootstrap=10000):
    return mean_ci(
        [r["weighted_accuracy"] for rs in results['all_instance_results'] for r in rs],
        ci=ci,
        n_bootstrap=n_bootstrap
    )


def mean_ci(values, ci=0.95, n_bootstrap=10000):
    return bootstrap_ci(
        values, metric=np.mean, ci=ci, n_bootstrap=n_bootstrap
    )
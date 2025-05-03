from genlm.control.sampler import EagerSetSampler, SetTokenSampler


class ImproperlyWeightedSetTokenSampler(SetTokenSampler):
    async def sample(self, context):
        x, _, logp = await super().sample(context)
        return x, 0, logp


def improperly_weighted_eager_token_sampler(llm, bool_cfg):
    return ImproperlyWeightedSetTokenSampler(EagerSetSampler(llm, bool_cfg))

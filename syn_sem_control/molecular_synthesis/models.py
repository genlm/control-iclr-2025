from syn_sem_control.util import improperly_weighted_eager_token_sampler
from genlm.control.sampler import DirectTokenSampler, eager_token_sampler
from genlm.eval.domains.molecular_synthesis.model import MolecularSynthesisModel, PartialSMILES


class BaseMolecularSynthesisModel(MolecularSynthesisModel):
    def get_sampler_cache_key(self, instance):
        return "molecular_synthesis"

    def fast_potential(self, instance):
        return self.bool_cfg

    def expensive_potential(self, instance):
        return PartialSMILES().coerce(self.llm, f=b"".join)
    
    def metadata(self):
        md = super().metadata() 
        md["model"] = self.__class__.__name__
        return md
    

class BaseLM(BaseMolecularSynthesisModel):
    def make_critic(self, instance):
        return

    def make_sampler(self, instance):
        return DirectTokenSampler(self.llm) 
    

class GrammarOnlyProperlyWeighted(BaseMolecularSynthesisModel):
    def make_critic(self, instance):
        return 
    
    def make_sampler(self, instance):
        return eager_token_sampler(self.llm, self.fast_potential(instance))
    

class GrammarOnlyImproperlyWeighted(BaseMolecularSynthesisModel):
    def make_critic(self, instance):
        return self.expensive_potential(instance)

    def make_sampler(self, instance):
        return improperly_weighted_eager_token_sampler(
            self.llm, self.fast_potential(instance)
        )
    

class FullModelImproperlyWeighted(BaseMolecularSynthesisModel):
    def make_critic(self, instance):
        return self.expensive_potential(instance)
    
    def make_sampler(self, instance):
        return improperly_weighted_eager_token_sampler(
            self.llm, self.fast_potential(instance)
        )
    

class FullModel(BaseMolecularSynthesisModel):
    def make_critic(self, instance):
        return self.expensive_potential(instance)
    
    def make_sampler(self, instance):
        return eager_token_sampler(
            self.llm, self.fast_potential(instance)
        )
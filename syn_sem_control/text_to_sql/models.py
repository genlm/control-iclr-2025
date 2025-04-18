import os
from syn_sem_control.util import improperly_weighted_eager_token_sampler
from genlm.control.sampler import DirectTokenSampler, eager_token_sampler
from genlm.eval.domains.spider import SpiderModel
from genlm.eval.domains.spider.table_column_potential import SpiderTableColumnVerifier


class BaseSpiderModel(SpiderModel):
    def get_sampler_cache_key(self, instance):
        return instance.schema_name

    def fast_potential(self, instance):
        return self.bool_cfg(instance.schema_name)

    def expensive_potential(self, instance):
        grammar_path = os.path.join(self.grammar_dir, instance.schema_name + ".lark")
        
        if not os.path.exists(grammar_path):
            raise FileNotFoundError(
                f"Grammar file not found for schema {instance.schema_name}"
            )
        
        with open(grammar_path, "r") as f:
            grammar = f.read()
        
        return SpiderTableColumnVerifier(
            grammar=grammar, tables=instance.tables
        ).coerce(self.llm, f=b''.join)
    
    def metadata(self):
        md = super().metadata() 
        md["model"] = self.__class__.__name__
        return md
    

class BaseLM(BaseSpiderModel):
    def make_critic(self, instance):
        return

    def make_sampler(self, instance):
        return DirectTokenSampler(self.llm) 
    

class GrammarOnlyProperlyWeighted(BaseSpiderModel):
    def make_critic(self, instance):
        return 
    
    def make_sampler(self, instance):
        return eager_token_sampler(self.llm, self.fast_potential(instance))
    

class GrammarOnlyImproperlyWeighted(BaseSpiderModel):
    def make_critic(self, instance):
        return self.expensive_potential(instance)

    def make_sampler(self, instance):
        return improperly_weighted_eager_token_sampler(
            self.llm, self.fast_potential(instance)
        )
    

class FullModelImproperlyWeighted(BaseSpiderModel):
    def make_critic(self, instance):
        return self.expensive_potential(instance)
    
    def make_sampler(self, instance):
        return improperly_weighted_eager_token_sampler(
            self.llm, self.fast_potential(instance)
        )
    

class FullModel(BaseSpiderModel):
    def make_critic(self, instance):
        return self.expensive_potential(instance)
    
    def make_sampler(self, instance):
        return eager_token_sampler(
            self.llm, self.fast_potential(instance)
        )
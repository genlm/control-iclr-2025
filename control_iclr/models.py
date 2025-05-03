import os
import time
from functools import cached_property
from abc import ABC, abstractmethod
from genlm.eval import ModelOutput, ModelResponse
from genlm.control import direct_token_sampler, eager_token_sampler, PromptedLLM
from .util import improperly_weighted_eager_token_sampler
from collections import OrderedDict


class Model(ABC):
    """Base abstract class for all models.

    Provides common functionality for language model inference with different sampling strategies
    and potential functions.

    Args:
        lm_name: Name of the language model
        lm_args: Arguments for the language model
        prompt_formatter: Function to format input prompts. Takes a tokenizer and an instance and returns a list of token ids.
        potential_factory: Factory class for creating potential functions
        n_particles (int): Number of particles for SMC sampling
        max_tokens (int): Maximum number of tokens to generate
        ess_threshold (float): Effective sample size threshold for resampling
        resampling_method (str): Method to use for resampling
        cache_key_fn: Function to generate cache key for instances. Takes an instance and returns a cache key. If None, caching is disabled.
        max_cache_size: Maximum number of cached samplers
        eos_token_factory: Function to generate EOS tokens for the language model. Takes a language model and returns a list of EOS tokens. If None, the language model's EOS tokens are used.
    """

    def __init__(
        self,
        lm_name,
        prompt_formatter,
        potential_factory,
        n_particles: int,
        max_tokens: int,
        ess_threshold: float,
        resampling_method: str,
        cache_key_fn=None,
        max_cache_size: int = 1,
        lm_args=None,
        eos_token_factory=None,
    ):
        self.lm_name = lm_name
        self.lm_args = lm_args or {}
        self.prompt_formatter = prompt_formatter
        self.potential_factory = potential_factory
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.resampling_method = resampling_method
        self.max_tokens = max_tokens
        self.cache_key_fn = cache_key_fn
        self.max_cache_size = max_cache_size
        self._sampler_cache = OrderedDict()
        self.eos_token_factory = eos_token_factory

    @cached_property
    def llm(self):
        llm = PromptedLLM.from_name(self.lm_name, **self.lm_args)
        if self.eos_token_factory is None:
            return llm
        else:
            eos_tokens = self.eos_token_factory(llm)
            return llm.spawn_new_eos(eos_tokens)

    def get_cache_key(self, instance):
        """Get cache key for an instance if caching is enabled."""
        if self.cache_key_fn is None:
            return None
        return self.cache_key_fn(instance)

    def make_sampler(self, instance):
        """Create or retrieve cached sampler for the instance.

        Args:
            instance (genlm.eval.Instance): Input instance to create sampler for

        Returns:
            genlm.control.Sampler: A sampler for the model
        """
        cache_key = self.get_cache_key(instance)
        if cache_key is not None:
            if cache_key in self._sampler_cache:
                self._sampler_cache.move_to_end(cache_key)
                return self._sampler_cache[cache_key]

            sampler = self._make_sampler(instance)

            if len(self._sampler_cache) >= self.max_cache_size:
                self._sampler_cache.popitem(last=False)

            self._sampler_cache[cache_key] = sampler
            return sampler

        return self._make_sampler(instance)

    @abstractmethod
    def _make_sampler(self, instance):
        """Abstract method to create a new sampler for the model.

        Args:
            instance (genlm.eval.Instance): Input instance to create sampler for

        Returns:
            (genlm.control.TokenSampler): A sampler for the model
        """
        pass

    # Default implementation returns None
    def make_critic(self, instance):
        """Default implementation returns None.

        Subclasses should override this method to provide a concrete
        implementation of the critic.
        """
        return None

    async def __call__(self, instance, output_dir, replicate):
        """Asynchronous method to execute the model.

        Args:
            instance (genlm.eval.Instance): Input instance to execute the model on
            output_dir (str): Directory to save output files
            replicate (int): Replicate number for output files

        Returns:
            (genlm.eval.ModelOutput): Output of the model
        """
        self.llm.prompt_ids = self.prompt_formatter(self.llm.model.tokenizer, instance)
        sampler = self.make_sampler(instance)
        critic = self.make_critic(instance)
        if critic is not None:
            critic.coerce(self.llm, f=b"".join)

        json_path = os.path.join(
            output_dir, f"{instance.instance_id}-{replicate}-record.json"
        )

        time_start = time.time()
        sequences = await sampler.smc(
            n_particles=self.n_particles,
            ess_threshold=self.ess_threshold,
            max_tokens=self.max_tokens,
            resampling_method=self.resampling_method,
            critic=critic,
            json_path=json_path,
        )
        time_end = time.time()

        return ModelOutput(
            responses=[
                ModelResponse(response=sequence, weight=prob)
                for sequence, prob in sequences.decoded_posterior.items()
            ],
            runtime=time_end - time_start,
        )


class PotentialFactory(ABC):
    """Abstract factory for creating potential functions.

    Defines interface for creating both fast (token-level) and expensive (sequence-level)
    potential functions used in controlled text generation.
    """

    @abstractmethod
    def get_fast_potential(self, instance):
        """Creates a fast potential function for token-level scoring.

        Args:
            instance (genlm.eval.Instance): Input instance to create potential function for

        Returns:
            genlm.control.Potential: A potential function for token-level scoring
        """
        pass

    @abstractmethod
    def get_expensive_potential(self, instance):
        """Creates an expensive potential function for sequence-level scoring.

        Args:
            instance (genlm.eval.Instance): Input instance to create potential function for

        Returns:
            genlm.control.Potential: A potential function for sequence-level scoring
        """
        pass


class BaseLM(Model):
    """Language model baseline."""

    def _make_sampler(self, instance):
        return direct_token_sampler(self.llm)


class FastBase(Model, ABC):
    """Base class for models using fast potential functions."""

    @property
    @abstractmethod
    def sampler_cls(self):
        pass

    def _make_sampler(self, instance):
        return self.sampler_cls(
            self.llm, self.potential_factory.get_fast_potential(instance)
        )


class FastProperlyWeighted(FastBase):
    """Properly weighted model with fast potential function."""

    @property
    def sampler_cls(self):
        return eager_token_sampler


class FastImproperlyWeighted(FastBase):
    """Improperly weighted model with fast potential function."""

    @property
    def sampler_cls(self):
        return improperly_weighted_eager_token_sampler


class ExpensivePotentialMixin:
    """Mixin class to add expensive potential functions."""

    def make_critic(self, instance):
        return self.potential_factory.get_expensive_potential(instance)


class FullProperlyWeighted(FastProperlyWeighted, ExpensivePotentialMixin):
    """Complete model using properly weighted sampling and both potentials."""

    pass


class FullImproperlyWeighted(FastImproperlyWeighted, ExpensivePotentialMixin):
    """Complete model using improperly weighted sampling and both potentials."""

    pass

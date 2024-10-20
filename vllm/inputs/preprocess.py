import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from typing_extensions import assert_never

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.utils import print_warning_once

from .data import DecoderOnlyInputs, PromptType, SingletonPrompt
from .parse import parse_singleton_prompt

logger = init_logger(__name__)

PromptComponents = Tuple[Optional[str], List[int],
                         Optional[Dict[str, Any]]]
DecoderPromptComponents = Tuple[Optional[str], Optional[List[int]],
                                Optional[Dict[str, Any]]]


class InputPreprocessor:
    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: Optional[BaseTokenizerGroup],
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.tokenizer = tokenizer

    def get_tokenizer_group(self) -> BaseTokenizerGroup:
        if self.tokenizer is None:
            raise ValueError("You cannot pass text prompts when "
                             "`skip_tokenizer_init` is True")

        return self.tokenizer

    def get_bos_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning("Using None for BOS token id because tokenizer "
                           "is not initialized")
            return None

        return self.tokenizer.tokenizer.bos_token_id

    def get_eos_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            logger.warning("Using None for EOS token id because tokenizer "
                           "is not initialized")
            return None

        return self.tokenizer.tokenizer.eos_token_id

    def get_decoder_start_token_id(self) -> Optional[int]:
        '''
        Obtain the decoder start token id employed by an encoder/decoder
        model. Returns None for non-encoder/decoder models or if the
        model config is unavailable.
        '''

        if (self.model_config is None or self.model_config.hf_config is None):
            print_warning_once("Using None for decoder start token id because "
                               "model config is not available.")
            return None

        dec_start_token_id = getattr(self.model_config.hf_config,
                                     'decoder_start_token_id', None)
        if dec_start_token_id is None:
            print_warning_once("Falling back on <BOS> for decoder start token "
                               "id because decoder start token id is not "
                               "available.")
            dec_start_token_id = self.get_bos_token_id()

        return dec_start_token_id

    def _prepare_decoder_input_ids_for_generation(
        self,
        decoder_input_ids: Optional[List[int]],
        force_bos: bool = True,
    ) -> List[int]:
        """
        Prepares `decoder_input_ids` for generation with encoder-decoder models.

        Based on

        https://github.com/huggingface/transformers/blob/
        4037a2b5b1278736e566aec12e169100275545ea/
        src/transformers/generation/utils.py

        specifically GenerationMixin._prepare_decoder_input_ids_for_generation()

        Arguments:

        * decoder_input_ids: input token ids to preprocess

        Returns:

        * Processed token list
        """

        decoder_start_token_id = self.get_decoder_start_token_id()
        assert decoder_start_token_id is not None

        if decoder_input_ids is None:
            # no decoder prompt input ->
            # use decoder_start_token_id as decoder_input_ids
            decoder_input_ids = self._get_default_enc_dec_decoder_prompt()

        if force_bos and (len(decoder_input_ids) == 0
                          or decoder_input_ids[0] != decoder_start_token_id):
            decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

        return decoder_input_ids

    def _tokenize_prompt(self, prompt: str, request_id: str) -> List[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        return self.get_tokenizer_group().encode(request_id=request_id, prompt=prompt)

    async def _tokenize_prompt_async(self, prompt: str, request_id: str) -> List[int]:
        """Async version of :meth:`_tokenize_prompt`."""
        tokenizer = self.get_tokenizer_group()

        return await tokenizer.encode_async(request_id=request_id,
                                            prompt=prompt)

    def _extract_prompt_components(
        self,
        prompt: SingletonPrompt,
        request_id: str) -> PromptComponents:
        '''
        Extract the components of any single encoder or decoder input prompt.

        Arguments:

        * request_id
        * prompt: single encoder or decoder input prompt

        Returns:

        * prompt
        * prompt_token_ids
        '''

        parsed = parse_singleton_prompt(prompt)

        if parsed["type"] == "str":
            prompt_text = parsed["content"]
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                request_id=request_id,
            )
        elif parsed["type"] == "tokens":
            prompt_text = None
            prompt_token_ids = parsed["content"]["prompt_token_ids"]
        elif parsed["type"] == "text":
            prompt_text = parsed["content"]["prompt"]
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                request_id=request_id,
            )
        else:
            assert_never(parsed)

        return (prompt_text, prompt_token_ids)

    async def _extract_prompt_components_async(
        self,
        prompt: SingletonPrompt,
        request_id: str,
    ) -> PromptComponents:
        """Async version of :meth:`_extract_prompt_components`."""
        parsed = parse_singleton_prompt(prompt)

        if parsed["type"] == "str":
            prompt_text = parsed["content"]
            prompt_token_ids = await self._tokenize_prompt_async(
                prompt_text,
                request_id=request_id,
            )
        elif parsed["type"] == "tokens":
            prompt_text = None
            prompt_token_ids = parsed["content"]["prompt_token_ids"]
        elif parsed["type"] == "text":
            prompt_text = parsed["content"]["prompt"]
            prompt_token_ids = await self._tokenize_prompt_async(
                prompt_text,
                request_id=request_id,
            )
        else:
            assert_never(parsed)

        return (prompt_text, prompt_token_ids)

    def _build_decoder_only_llm_inputs(
        self,
        prompt_comps: PromptComponents,
    ) -> DecoderOnlyInputs:
        (prompt, prompt_token_ids) = prompt_comps

        return DecoderOnlyInputs(prompt_token_ids=prompt_token_ids,
                                 prompt=prompt)

    def _process_decoder_only_prompt(
        self,
        prompt: SingletonPrompt,
        request_id: str,
    ) -> DecoderOnlyInputs:
        '''
        For decoder-only models:
        Process an input prompt into an :class:`DecoderOnlyInputs` instance.

        Arguments:

        * prompt: input prompt
        * request_id

        Returns:

        * :class:`DecoderOnlyInputs` instance
        '''

        prompt_comps = self._extract_prompt_components(
            prompt,
            request_id=request_id,
        )

        return self._build_decoder_only_llm_inputs(prompt_comps)

    async def _process_decoder_only_prompt_async(
        self,
        prompt: SingletonPrompt,
        request_id: str,
    ) -> DecoderOnlyInputs:
        """Async version of :meth:`_process_decoder_only_prompt`."""
        prompt_comps = await self._extract_prompt_components_async(
            prompt,
            request_id=request_id,
        )

        return self._build_decoder_only_llm_inputs(
            prompt_comps,
        )

    def preprocess(
        self,
        prompt: PromptType,
        request_id: str,
    ) -> DecoderOnlyInputs:
        """Preprocess the input prompt."""
        # Decoder-only operation
        return self._process_decoder_only_prompt(
            prompt,
            request_id=request_id,
        )

    async def preprocess_async(
        self,
        prompt: PromptType,
        request_id: str,
    ) -> DecoderOnlyInputs:
        """Async version of :meth:`preprocess`."""
        # Decoder-only operation
        return await self._process_decoder_only_prompt_async(
            prompt,
            request_id=request_id,
        )

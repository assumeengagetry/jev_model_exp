def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], is_init_check: bool = False
    ) -> str:
        """
        Check that the `attn_implementation` exists and is supported by the models, and try to get the kernel from hub if
        it matches hf kernels pattern.

        Args:
            attn_implementation (`str` or `None`):
                The attention implementation to check for existence/validity.
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, and/or if the model uses
                BetterTransformer, which are only available later after __init__. This allows to raise proper exceptions early
                before instantiating the full models if we know that the model does not support the requested attention.

        Returns:
            `str`: The final attention implementation to use, including potential fallbacks from sdpa to eager, or from
            None to sdpa (to potentially eager).
        """
        applicable_attn_implementation = attn_implementation

        # If FA not installed, do not fail but use kernels instead
        if (
            attn_implementation is not None
            and attn_implementation.startswith("flash_attention")
            and self._supports_flash_attn
            and not (is_flash_attn_2_available() or is_flash_attn_3_available())
            and is_kernels_available()
        ):
            if attn_implementation.endswith("2"):
                applicable_attn_implementation = "kernels-community/flash-attn"
            else:
                applicable_attn_implementation = "kernels-community/vllm-flash-attn3"

        if is_kernel(applicable_attn_implementation):
            try:
                load_and_register_kernel(applicable_attn_implementation)
                # log that we used kernel fallback if successful
                if attn_implementation.startswith("flash_attention"):
                    logger.warning_once(
                        f"You do not have `flash_attn` installed, using `{applicable_attn_implementation}` "
                        "from the `kernels` library instead!"
                    )
            except Exception as e:
                # raise the proper exception for requested flash attention
                if attn_implementation.startswith("flash_attention"):
                    if attn_implementation.endswith("2"):
                        self._flash_attn_2_can_dispatch()
                    else:
                        self._flash_attn_3_can_dispatch()

                # error properly out if a kernel was specifically requested
                raise e
        else:
            applicable_attn_implementation = self.get_correct_attn_implementation(
                applicable_attn_implementation, is_init_check
            )
            # preload flash attention here to allow compile with fullgraph
            if applicable_attn_implementation.startswith("flash_attention"):
                lazy_import_flash_attention(applicable_attn_implementation, force_import=True)

        return applicable_attn_implementation

def get_correct_attn_implementation(self, requested_attention: Optional[str], is_init_check: bool = False) -> str:
        applicable_attention = "sdpa" if requested_attention is None else requested_attention

        if applicable_attention not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
            message = (
                f'Specified `attn_implementation="{applicable_attention}"` is not supported. The only possible arguments are '
                '`attn_implementation="eager"`'
            )
            # check `supports_flash_attn_2` for BC with custom code. TODO: remove after a few releases
            if self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False):
                message += ', `"attn_implementation=flash_attention_3"`, `"attn_implementation=flash_attention_2"`'
            if self._supports_sdpa:
                message += ', `"attn_implementation=sdpa"'
            if self._supports_flex_attn:
                message += ', `"attn_implementation=flex_attention"`'
            raise ValueError(message + ".")

        # Perform relevant checks
        if applicable_attention == "flash_attention_2":
            self._flash_attn_2_can_dispatch(is_init_check)
        elif applicable_attention == "flash_attention_3":
            self._flash_attn_3_can_dispatch(is_init_check)
        elif applicable_attention == "flex_attention":
            self._flex_attn_can_dispatch(is_init_check)
        elif applicable_attention == "sdpa":
            # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
            try:
                self._sdpa_can_dispatch(is_init_check)
            except (ValueError, ImportError) as e:
                if requested_attention == "sdpa":
                    raise e
                applicable_attention = "eager"

        return applicable_attention
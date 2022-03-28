from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Union

from flatten_dict import flatten, unflatten
from kedro.config import TemplatedConfigLoader


class CustomTemplatedConfigLoader(TemplatedConfigLoader):
    def __init__(
            self,
            conf_paths: Union[str, Iterable[str]],
            *,
            globals_pattern: Optional[str] = None,
            globals_dict: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate a ``CustomTemplatedConfigLoader``.
        Args:
            conf_paths: Non-empty path or list of paths to configuration
                directories.
            globals_pattern: Optional keyword-only argument specifying a glob
                pattern. Files that match the pattern will be loaded as a
                formatting dictionary.
            globals_dict: Optional keyword-only argument specifying a formatting
                dictionary. This dictionary will get merged with the globals dictionary
                obtained from the globals_pattern. In case of duplicate keys, the
                ``globals_dict`` keys take precedence.
        This class allows to overwrite nested parametes with globals_dict specifing
        nested keys as nested.param.abc, example:
        globals_dict = {
            'nested': {
                'param': 1
            }
        }
        """

        super(TemplatedConfigLoader, self).__init__(conf_paths)

        self._arg_dict = (
            super(TemplatedConfigLoader, self).get(globals_pattern)
            if globals_pattern
            else {}
        )
        globals_dict = deepcopy(globals_dict) or {}
        flat_args_dict = flatten(self._arg_dict, reducer="dot")
        flat_args_dict = {**flat_args_dict, **globals_dict}
        self._arg_dict = unflatten(flat_args_dict, splitter="dot")

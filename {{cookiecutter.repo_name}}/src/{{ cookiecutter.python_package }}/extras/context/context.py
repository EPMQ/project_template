from typing import Dict, Any

from flatten_dict import flatten, unflatten
from kedro.config import MissingConfigException
from kedro.framework.context import KedroContext
from warnings import warn


class CustomContext(KedroContext):
    @property
    def params(self) -> Dict[str, Any]:
        """Read-only property referring to Kedro's parameters for this context.
        This context allow passing nested extra parameters to the cli, updating
        the parameters defined in `parameters.yml` accordingly.
        Example:
            Considering the parameters in `parameters.yml` to be:
                nested:
                    param_1:
                        param: 1
                    param_2:
                        param: 2
            running `kedro run --params nested.param_1.param:5` will result
            in the following parameters:
                {
                    'nested': {
                        'param_1': {
                            'param': 5
                        },
                        'param_2': {
                            'param': 2
                        },
                    }
                }
        Returns:
            Parameters defined in `parameters.yml` with the addition of any
                extra parameters passed at initialization.
        """
        try:
            # '**/parameters*' reads modular pipeline configs
            params = self.config_loader.get(
                "parameters*", "parameters*/**", "**/parameters*"
            )
        except MissingConfigException as exc:
            warn(f"Parameters not found in your Kedro project config.\n{str(exc)}")
            params = {}
        flat_params = flatten(params, reducer="dot")
        flat_params.update(self._extra_params or {})
        params = unflatten(flat_params, splitter="dot")
        return params

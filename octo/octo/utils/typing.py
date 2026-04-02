from typing import Any, Mapping, Sequence, Union

import jax

# 兼容新旧 JAX：老版本公开 `jax.random.KeyArray`，新版本已不再暴露该别名。
try:
    PRNGKey = jax.random.KeyArray
except AttributeError:
    PRNGKey = jax.Array
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike

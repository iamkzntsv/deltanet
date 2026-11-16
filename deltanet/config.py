from omegaconf import OmegaConf

OmegaConf.register_resolver("frac", lambda n, f: int(n * f), replace=True)
OmegaConf.register_resolver("div", lambda a, b: a // b, replace=True)
OmegaConf.register_resolver("mul", lambda a, b: a * b, replace=True)

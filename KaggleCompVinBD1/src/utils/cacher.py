from pathlib import Path
import pickle

from src.utils.paths import CACHE_DIR
from src import config, log


# Decorator for caching funcs
def cache(**outer_kwargs):

    prefix = outer_kwargs.get("prefix", "")

    def inner(func):

        def wrapper(*args, **kwargs):
            cache_path = Path(f"{str(CACHE_DIR)}/{prefix}{func.__name__}_{config.__hash__()}")

            if config.image_size:
                cache_path = Path(str(cache_path) + f'{config.image_size}' + ".pkl")

            use_cache = config.use_cache

            if use_cache and cache_path.exists():
                with open(cache_path, mode='rb') as f:
                    out = pickle.load(f)

                log.info(f"Cache used for {func.__name__} found in {cache_path}")

                return out
            else:
                out = func(*args, **kwargs)
                if use_cache:
                    with open(cache_path, mode='wb') as f:
                        pickle.dump(out, f)

                    log.info(f"Cached {func.__name__} output to {cache_path}")

                return out

        return wrapper

    return inner

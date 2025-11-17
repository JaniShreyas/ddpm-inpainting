from .linear import linear_noise_schedule
from .cosine import cosine_noise_schedule

SCHEDULE_REGISTRY = {
    "linear": linear_noise_schedule,
    "cosine": cosine_noise_schedule
}

def get_noise_schedule(schedule_config):
    name = schedule_config.get("name")
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Schedule '{name}' not found. Available schedules: {list(SCHEDULE_REGISTRY.keys())}")
    
    builder_fn = SCHEDULE_REGISTRY[name]
    return builder_fn(**schedule_config)
import time
import statistics
from util.settings_handler import Settings

# "module": [...measurements]
measurements = {}

def start_time() -> float:
    return time.time()


def elapsed_time(start_time: float) -> float:
    return time.time() - start_time


def elapsed_minutes(start_time: float) -> str:
    """ Minutes elapsed since start time (2 decimals) """
    t_delta = time.time() - start_time
    return round(t_delta / 60, 2)


def elapsed_seconds(start_time: float, module: str) -> str:
    """ Minutes elapsed since start time (2 decimals) """
    t_delta = time.time() - start_time

    # Add module measurement to dict
    if module in measurements:
        measurements[module].append(t_delta)
    else:
        measurements[module] = [t_delta]

    return round(t_delta, 2)


def print_elapsed_seconds(settings: Settings, start_time: int, module: str) -> None:
    if settings.measure_time:
        # Only print if setting is on
        print(f"{module} finished in {elapsed_seconds(start_time, module)} seconds")


def print_average_measurements(settings: Settings) -> None:
    if settings.measure_time == False:
        return
    
    for module, m_list in measurements.items():
        mean = statistics.fmean(m_list)
        print(f"{module} mean time {round(mean, 2)} seconds {m_list}")


def print_average_measurements_skip_first(settings: Settings) -> None:
    if settings.measure_time == False:
        return
    
    for module, m_list in measurements.items():
        if len(m_list) == 1:
            mean = m_list[0]
        else:
            skipped_list = m_list[1:-1]
            mean = statistics.fmean(skipped_list)
        
        print(f"{module} (skip first) mean time {round(mean, 2)} seconds {m_list}")
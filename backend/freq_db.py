from backend.freq_db import FREQ_DB

def match_frequency(freq_mhz):
    for band in FREQ_DB:
        low, high = band["range"]
        if low <= freq_mhz <= high:
            return {
                "label": band["label"],
                "modulation": band["mod"],
                "usage": band["usage"]
            }
    return {
        "label": "Неизвестно",
        "modulation": "?",
        "usage": "Не удалось определить диапазон"
    }
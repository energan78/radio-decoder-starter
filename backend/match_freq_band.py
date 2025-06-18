from backend.freq_db import FREQ_DB

def match_frequency(freq):
    for band in FREQ_DB:
        low = band["start"]
        high = band["end"]
        if low <= freq <= high:
            return band
    return None
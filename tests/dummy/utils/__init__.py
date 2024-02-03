import uuid


def select_random_port(max_number: int = 2**16) -> int:
    seed = str(uuid.uuid4())
    seed = seed.replace("-", "")
    seed = int(seed, 16)

    return seed % max_number

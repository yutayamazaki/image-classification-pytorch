import csv
from typing import List


def read_csv(path: str) -> List[List[str]]:
    with open(path, 'r') as f:
        reader = csv.reader(f)
        return [row for row in reader]


def dump_csv(path: str, data: List[List[str]]):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def append_row(path: str, data: List[str]):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

import numpy as np
from typing import List
from scheduler.job import Job


def first_fit(items: List[np.array]) -> List[List[np.array]]:
    # First Fit Vector Bin Packing
    # All input vectors must have the same dimension
    # All bins have capacity <1,1,1,...,1>
    bins = [[]]

    for item in items:
        can_fit = False
        for bin_ in bins:
            if np.all(np.add.reduce(bin_) + item <= np.ones(len(item))):
                bin_.append(item)
                can_fit = True
                break

        if not can_fit:
            bins.append([item])

    return bins


def first_fit_job(jobs: List[Job]) -> List[List[Job]]:
    bins = [[]]

    for job in jobs:
        can_fit = False
        for bin_ in bins:
            if np.all(np.add.reduce([j.d for j in bin_]) + job.d <= np.ones(np.size(job.d))):
                bin_.append(job)
                can_fit = True
                break

        if not can_fit:
            bins.append([job])

    return bins


def best_fit(items: List[np.array]) -> List[List[np.array]]:
    # Best Fit Vector Bin Packing. Maximum load is defined as maximum over all dimensions
    # All input vectors must have the same dimension
    # All bins have capacity <1,1,1,...,1>
    bins = [[]]

    for item in items:
        loads = [np.max(np.add.reduce(bin_) + item) for bin_ in bins]
        best_bin = bins[np.argmin(loads)]
        if np.all(best_bin) <= np.ones(len(item)):
            best_bin.append(item)
        else:
            bins.append([item])

    return bins

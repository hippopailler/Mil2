
import os
from pipeline.metrics.compute_scores import compute_scores
from pipeline.metrics.calculate_average import compute_weighted_average
from pipeline_utils import plot_comparative_results


path = "cohort2/cross_validation/mil/os_months_24/classification/cross_validation/hypothesis_driven/pyrad-noimp/rwd/seed_0"
compute_scores(path, task="classification")
compute_weighted_average(path, task="classification")
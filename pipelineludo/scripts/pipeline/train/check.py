import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pipeline.metrics.calculate_average import compute_weighted_average

paths = [
   #  "/Users/ludole/Desktop/PHD/matteMIL/cross_validation/mil/ORR",
    "/Users/ludole/Desktop/PHD/matteMIL/cross_validation/mil/OS_MONTHS",
   #  "/Users/ludole/Desktop/PHD/matteMIL/cross_validation/mil/os_months_24"
]

for path in paths:
    if os.path.exists(path):
        compute_weighted_average(path, "survival")
    else:
        print(f"❌ Path not found: {path}")
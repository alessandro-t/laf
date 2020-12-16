#!/bin/bash
# x can be one among {count, sum, max, mean, min, max_minus_min
#                     inv_count, median, min_over_max, std,
#                     skew, kurtosis}
x=count;
python generate_dataset.py --label $x --seed 42 --bias;
python generate_dataset.py --label $x --seed 42;
python train_plain.py --model laf --units 9 --label $x --seed 42  --run 0;
python train.py --model laf --units 9 --label $x --seed 42  --run 0;

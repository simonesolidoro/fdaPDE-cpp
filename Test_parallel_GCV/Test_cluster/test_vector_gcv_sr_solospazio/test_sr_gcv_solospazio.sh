#!/bin/bash

#n_threads=(1 2 4) #(1 2 4 8 12 16) # 1 lasciato per vedere overhead threadpool rispetto seq
output_file="test_sr_gcv_solospazio.txt"
./test_sr_gcv_solospazio >> "$output_file"


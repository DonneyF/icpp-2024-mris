#!/bin/bash

DATA_LOCATION="./experiments/data"

NUM_RESOURCES="${1:-20}"

if [ ! -d $DATA_LOCATION ]; then
  mkdir $DATA_LOCATION
fi

# Download Azure Dataset
if [ ! -f $DATA_LOCATION/packing_trace_zone_a_v1.sqlite ]; then
  wget --no-verbose https://icpp-2024-mris.s3.us-west-001.backblazeb2.com/packing_trace_zone_a_v1.sqlite -P $DATA_LOCATION
fi
if [ -f $DATA_LOCATION/AzurePackingTraceV1.zip ]; then
  unzip $DATA_LOCATION/AzurePackingTraceV1.zip -d $DATA_LOCATION
fi

# Generate the datasets
export PYTHONPATH="${PYTHONPATH}:${PWD%/*}/scheduler"

echo "Preprocessing datasets (up to $((NUM_RESOURCES - 4)) datasets)"
python3 << END
from experiments.tools import DataGenerator
dataset = DataGenerator(N=4096000, M=20, R=$NUM_RESOURCES, dataset="azure_packing_2020", data_location="$DATA_LOCATION")
END

#!/bin/bash

while true; do
    echo "🔁 Starting TOF inference script..."

    # Start and kill after timeout (e.g., 55 minutes)
    timeout 55m python TOFMapper/large_scale_inference.py -i Images -u data/utm_grid/Sachsen_Grid_ETRS89-UTM32_1km.gpkg -c TOFMapper/config/tof/ftunetformer.py -o results -ps 1024 -b 1

    echo "🕒 Script finished or timed out. Restarting after short pause..."
    sleep 5
done
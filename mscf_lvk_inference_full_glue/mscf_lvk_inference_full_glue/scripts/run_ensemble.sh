#!/bin/bash
# Population ensemble study: 4 SNR levels x 30 noise seeds each
# Fixed sampler seed = 42 for reproducibility
# Noise seeds: 1000-1029

cd /home/thepadrevictor/mscf-lvk/mscf_lvk_inference_full_glue/mscf_lvk_inference_full_glue
source .venv/bin/activate

SAMPLER_SEED=42
NLIVE=300

# SNR levels to test
SNRS=(20 30 40 50)

# Launch all jobs
for SNR in "${SNRS[@]}"; do
    mkdir -p out_ensemble/snr${SNR}
    for NOISE_SEED in $(seq 1000 1029); do
        OUTDIR="out_ensemble/snr${SNR}/n${NOISE_SEED}"
        LOGFILE="out_ensemble/snr${SNR}/n${NOISE_SEED}.log"

        # Skip if already completed
        if [ -f "${OUTDIR}/injection_summary.json" ]; then
            echo "Skipping SNR=${SNR} noise_seed=${NOISE_SEED} (already complete)"
            continue
        fi

        echo "Launching SNR=${SNR} noise_seed=${NOISE_SEED}"
        nohup python scripts/test_h0_injection_gated.py \
            --snr ${SNR} \
            --noise-seed ${NOISE_SEED} \
            --seed ${SAMPLER_SEED} \
            --nlive ${NLIVE} \
            --outdir ${OUTDIR} \
            > ${LOGFILE} 2>&1 &

        # Small delay to avoid overwhelming the system
        sleep 0.1
    done
done

echo "All ensemble jobs launched!"
echo "Monitor with: ls out_ensemble/*/n*/injection_summary.json | wc -l"

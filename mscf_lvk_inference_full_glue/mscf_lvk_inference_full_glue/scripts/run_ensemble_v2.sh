#!/bin/bash
# Bulletproof Population Ensemble Study v2
# - Explicit injection params (Mf=67.8, chi=0.68, etc.)
# - 2 sampler seeds per noise seed (41, 42)
# - Higher nlive for borderline SNR (30/40)
# - CSV summary logging

cd /home/thepadrevictor/mscf-lvk/mscf_lvk_inference_full_glue/mscf_lvk_inference_full_glue
source .venv/bin/activate

OUTBASE="out_ensemble_v2"
mkdir -p ${OUTBASE}

# CSV header
CSV="${OUTBASE}/ensemble_results.csv"
echo "snr,noise_seed,sampler_seed,nlive,Mf_med,chi_med,f0_med,Mf_pass,chi_pass,f0_pass,overall_pass,logZ" > ${CSV}

# Fixed injection parameters (GW150914-like)
MF=67.8
CHI=0.68
T0=0.001

# Noise seeds
NOISE_SEEDS=$(seq 1000 1029)

# Sampler seeds
SAMPLER_SEEDS=(41 42)

# SNR levels with appropriate nlive
declare -A NLIVE_MAP
NLIVE_MAP[20]=300
NLIVE_MAP[30]=600
NLIVE_MAP[40]=600
NLIVE_MAP[50]=300

# Launch all jobs
for SNR in 20 30 40 50; do
    NLIVE=${NLIVE_MAP[$SNR]}
    mkdir -p ${OUTBASE}/snr${SNR}

    for NOISE_SEED in ${NOISE_SEEDS}; do
        for SAMPLER_SEED in "${SAMPLER_SEEDS[@]}"; do
            OUTDIR="${OUTBASE}/snr${SNR}/n${NOISE_SEED}_s${SAMPLER_SEED}"
            LOGFILE="${OUTBASE}/snr${SNR}/n${NOISE_SEED}_s${SAMPLER_SEED}.log"

            # Skip if already completed
            if [ -f "${OUTDIR}/injection_summary.json" ]; then
                echo "Skipping SNR=${SNR} noise=${NOISE_SEED} sampler=${SAMPLER_SEED} (done)"
                continue
            fi

            echo "Launching SNR=${SNR} noise=${NOISE_SEED} sampler=${SAMPLER_SEED} nlive=${NLIVE}"
            nohup python scripts/test_h0_injection_gated.py \
                --Mf ${MF} \
                --chi ${CHI} \
                --t0 ${T0} \
                --snr ${SNR} \
                --noise-seed ${NOISE_SEED} \
                --seed ${SAMPLER_SEED} \
                --nlive ${NLIVE} \
                --outdir ${OUTDIR} \
                > ${LOGFILE} 2>&1 &

            # Stagger launches
            sleep 0.2
        done
    done
done

echo ""
echo "All ensemble v2 jobs launched!"
echo "Total: 4 SNRs x 30 noise seeds x 2 sampler seeds = 240 jobs"
echo ""
echo "Monitor progress:"
echo "  ls ${OUTBASE}/*/n*/injection_summary.json | wc -l"
echo ""
echo "Aggregate results when done:"
echo "  python scripts/aggregate_ensemble.py --base-dir ${OUTBASE}"

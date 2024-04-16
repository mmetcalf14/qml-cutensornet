nf=50
ntr=200
r=2
df="elliptic_preproc.csv"

mkdir raw/
mkdir raw/gaussian
mkdir raw/quantum

# Run the classical kernel experiments
for s in 5 8 20 25 30 35; do
  python classical_main.py $nf $ntr $ntr $s $df
done

# Run the quantum kernel experiments
for s in 5 8 20 25 30 35; do
for d in 1 2 4 6; do
for g in 0.1 0.5 1.0; do
    sbatch slurm_scripts/4gpus.sh "GPU" $nf $r $g $d $ntr $ntr $s $df
done
done
done

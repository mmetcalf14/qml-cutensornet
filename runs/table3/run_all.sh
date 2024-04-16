nf=50
ntr=200
d=1
g=1.0
df="elliptic_preproc.csv"

mkdir raw/

# Run the quantum kernel experiments
for s in 5 8 20 25 30 35; do
for r in 2 4 8 12 16 20; do
    sbatch slurm_scripts/4gpus.sh "GPU" $nf $r $g $d $ntr $ntr $s $df
done
done

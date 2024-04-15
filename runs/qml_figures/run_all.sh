r=2
g=0.1
d=1
s=5
df="elliptic_preproc.csv"

mkdir raw/

for nf in 15 50 100 165; do
    sbatch slurm_scripts/4gpus.sh "GPU" $nf $r $g $d 150 150 $s $df
    sbatch slurm_scripts/4gpus.sh "GPU" $nf $r $g $d 750 750 $s $df
    sbatch slurm_scripts/32gpus.sh "GPU" $nf $r $g $d 3200 3200 $s $df
done

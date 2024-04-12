nf=165
r=2
g=0.1
d=1
s=5
df="elliptic_preproc.csv"

mkdir raw/

sbatch slurm_scripts/2gpus.sh "GPU" $nf $r $g $d 200 200 $s $df
sbatch slurm_scripts/4gpus.sh "GPU" $nf $r $g $d 400 400 $s $df
sbatch slurm_scripts/8gpus.sh "GPU" $nf $r $g $d 800 800 $s $df
sbatch slurm_scripts/16gpus.sh "GPU" $nf $r $g $d 1600 1600 $s $df
sbatch slurm_scripts/32gpus.sh "GPU" $nf $r $g $d 3200 3200 $s $df

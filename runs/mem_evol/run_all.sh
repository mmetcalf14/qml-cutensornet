nf=100
ntr=5
r=2
g=1.0
s=5
df="elliptic_preproc.csv"

mkdir raw/
mkdir raw/d6/
mkdir raw/d12/
cd ../..

for d in 6 12; do
for x in 0 1 2 3 4 5 6 7; do
    python main_track_mem.py "GPU" $nf $r $g $d $ntr $ntr $s $df $x 2> runs/mem_evol/raw/d$d/$x.out
done
done

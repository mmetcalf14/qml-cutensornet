nf=100
ntr=5
r=2
g=1.0
s=5
df="elliptic_preproc.csv"

mkdir raw/
mkdir raw/cpu/
mkdir raw/gpu/
cd ../..

for d in 2 4 6 8 10; do
    python main_no_test.py "CPU" $nf $r $g $d $ntr $ntr $s $df
done

mv train_Nf* runs/crossover/raw/cpu/

for d in 2 4 6 8 10 12; do
    python main_no_test.py "GPU" $nf $r $g $d $ntr $ntr $s $df
done

mv train_Nf* runs/crossover/raw/gpu/

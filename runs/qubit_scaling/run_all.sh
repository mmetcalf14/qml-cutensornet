ntr=5
d=6
r=2
s=5
df="elliptic_preproc.csv"

cd ../..

for nf in 30 60 90 120 150 165; do
for g in 0.1 0.5 1.0; do
    python main_no_test.py "GPU" $nf $r $g $d $ntr $ntr $s $df
done
done

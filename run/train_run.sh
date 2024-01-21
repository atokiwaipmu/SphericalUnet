
basedir=/gpfs02/work/akira.tokiwa/gpgpu/Github/SphericalUnet/run

# arguments sample
# target="HR"
# model="diffusion"
# transform_type="sigmoid"
# cond="concat"
# schedule="linear"
# ifmask=True
# order=8
# nmap=100
# batch=24
# device=5

# sample command
# bash train_run.sh HR diffusion smoothed 8 100 24 5

target=$1
model=$2
transform_type=$3
order=$4
nmap=$5
batch=$6
device=$7

# create a file name
fname="$model"_"$target"_"$transform_type"_"$order"_"$nmap"_"$batch"_"$device"

# duplicate the template file as fname.sh
cp $basedir/train_template.sh $basedir/used/$fname.sh

# replace the template file with the values
sed -i "2s/fname/$fname/" $basedir/used/$fname.sh
sed -i "4s/fname/$fname/" $basedir/used/$fname.sh
sed -i "5s/fname/$fname/" $basedir/used/$fname.sh
sed -i "16s/_device/$device/" $basedir/used/$fname.sh
sed -i "21s/_model/$model/" $basedir/used/$fname.sh
sed -i "21s/_target/$target/" $basedir/used/$fname.sh
sed -i "21s/_trans/$transform_type/" $basedir/used/$fname.sh
sed -i "21s/_order/$order/" $basedir/used/$fname.sh
sed -i "21s/_nmap/$nmap/" $basedir/used/$fname.sh
sed -i "21s/_batch/$batch/" $basedir/used/$fname.sh

# submit the job
sbatch $basedir/used/$fname.sh

# delete the file
# rm $fname.sh
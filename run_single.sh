dataset="EMnist-alpha0.1-ratio0.1"
alg=FedAvg

### shared parameters ###
lamda="1"
model="cnn"
local_epochs=20
batch_size=32
num_users=10
num_glob_iters=200
learning_rate=0.01
times=1

python3 main.py --dataset $dataset \
                --algorithm $alg \
                --batch_size $batch_size \
                --local_epochs $local_epochs \
                --num_users $num_users \
                --lamda $lamda \
                --model $model \
                --learning_rate $learning_rate \
                --num_glob_iters $num_glob_iters \
                --times $times \
                --K 1
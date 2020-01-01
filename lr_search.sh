arr=("1e-2" "3e-2" "5e-2" "7e-2" "9e-2" "1e-3" "3e-3" "5e-3" "7e-3" "9e-3" "1e-4" "3e-4" "5e-4" "7e-4" "9e-4" "1e-5" "3e-5" "5e-5" "7e-5" "9e-5")
for value in ${arr[@]}
do
  mpirun -np 16 python -m baselines.wgail.run_mujoco --d_stepsize d_stepsize $value
done
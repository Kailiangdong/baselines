d_stepsizes=("3.34e-2" "3.36e-2" "3.38e-2" "3.4e-2" "3.42e-2" "3.44e-2" "3.46e-2")
d_steps=("10" "20" "40" "60" "80" "100")
for d_stepsize in ${d_stepsizes[@]}
do

    for d_step in ${d_steps[@]}
    do
          mpirun -np 16 python -m baselines.wgail.run_mujoco --d_stepsize $d_stepsize --d_step $d_step
    done

done
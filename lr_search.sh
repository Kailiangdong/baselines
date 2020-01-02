d_stepsizes=("3.2e-2" "3.4e-2" "3.6e-2" "3.8e-2" "4e-2" "4.2e-2" "5.2e-3" "5.4e-3" "5.6e-3" "5.8e-3" "6e-3" "6.2e-3")
vf_stepsizes=("4e-2" "3e-4" "6e-3" "4e-1" "4e-5")
for d_stepsize in ${d_stepsizes[@]}
do

    for vf_stepsize in ${vf_stepsizes[@]}
    do
          mpirun -np 16 python -m baselines.wgail.run_mujoco --d_stepsize $d_stepsize --vf_stepsize $vf_stepsize
    done

done
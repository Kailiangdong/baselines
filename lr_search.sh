trap "exit" INT
d_stepsizes=("3.4e-2" "3.42e-2" "3.44e-2" "3.46e-2")
d_steps=("20" "40" "60" "80" "100")
clip_values=("1e0" "5e-1" "2e-1" "1e-1" "5e-2" "2e-2" "1e-2")
adversary_entcoeffs=("1e-2" "1e-3" "3e-4" "1e-4")
for d_stepsize in ${d_stepsizes[@]}
do

    for d_step in ${d_steps[@]}
    do

        for clip_value in ${clip_values[@]}
        do
            for adversary_entcoeff in ${adversary_entcoeffs[@]}
            do
                mpirun -np 16 python -m baselines.wgail.run_mujoco --d_stepsize $d_stepsize --d_step $d_step --adversary_entcoeff $adversary_entcoeff --clip_value $clip_value
            done
        done

    done

done
#放里面
#/home/huawei/Autonomous_Simulator/thesis/reference/w_gail/baselines/log/trpo_gail.transition_limitation_-1.Hopper_2020_01_03_14_56_41_.g_step_1.d_step_50.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0

#放外面
#/home/huawei/Autonomous_Simulator/thesis/reference/w_gail/baselines/log/trpo_gail.transition_limitation_-1.Hopper_2020_01_03_14_29_35_.g_step_1.d_step_50.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0
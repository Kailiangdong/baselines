trap "exit" INT
d_stepsizes=("1e-2" "1e-3" "1e-4")
vf_stepsizes=("1e-2" "1e-3" "1e-4" "1e-5")
gradient_penaltys=("1e1" "1e0" "1e-1" "1e-2")
adversary_entcoeffs=("1e-2" "1e-3" "1e-4")
for d_stepsize in ${d_stepsizes[@]}
do

    for gradient_penalty in ${gradient_penaltys[@]}
    do

        for vf_stepsize in ${vf_stepsizes[@]}
        do
            for adversary_entcoeff in ${adversary_entcoeffs[@]}
            do
                mpirun -np 16 python -m baselines.wgail.run_mujoco --d_stepsize $d_stepsize --gradient_penalty $gradient_penalty --adversary_entcoeff $adversary_entcoeff --vf_stepsize $vf_stepsize
            done
        done

    done

done
#放里面
#/home/huawei/Autonomous_Simulator/thesis/reference/w_gail/baselines/log/trpo_gail.transition_limitation_-1.Hopper_2020_01_03_14_56_41_.g_step_1.d_step_50.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0

#放外面
#/home/huawei/Autonomous_Simulator/thesis/reference/w_gail/baselines/log/trpo_gail.transition_limitation_-1.Hopper_2020_01_03_14_29_35_.g_step_1.d_step_50.policy_entcoeff_0.adversary_entcoeff_0.001.seed_0
import os
import numpy as np
import torch

from helper import (
    F,
    generate_data,
    train_test_split,
)

def main():

    out_dir      = "./Noise_5_Percent/tau_0.02/Training_Data_Less_Params"  
    prefix       = "data"                      
    num_files    = 10                             

    nsr_sd       = 0.05                     
    tau_c        = 0.02                        
    device       = torch.device("cuda:0" if torch.cuda.is_available()
                                else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype        = torch.float32
    alphas_train      = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=dtype, device=device)
    splits       = 10
    T_window     = 15              
    L_total      = splits + T_window - 1
    N_samples    = 2                 
    in_conds     = 20
    dt           = 0.1
    coarsen      = 100
    val_size     = 0.25

    nsr_var = float(nsr_sd ** 2)      
    theta   = float(1.0 / tau_c)      
    apply_ou_noise = (nsr_sd > 0.0)

   
    os.makedirs(out_dir, exist_ok=True)

    
    for rep in range(num_files):
    
        trajectories, params, indices = generate_data(
            alphas_train, F, L_total, T_window, N_samples,
            dt=dt, in_conds=in_conds, coarsening_factor=coarsen,
            nsr=nsr_var, theta=theta, burn_in_var=0,
            apply_ou_noise=apply_ou_noise, device=device, dtype=dtype,
            seed_ou=None    
        )

        (train_trajectories, train_params, train_indices), \
        (val_trajectories,   val_params,   val_indices) = train_test_split(
            trajectories, params, indices, val_size=val_size
        )

        train_trajectories_np = train_trajectories.detach().cpu().numpy()
        train_params_np       = train_params.detach().cpu().numpy()
        train_indices_np      = train_indices.detach().cpu().numpy()
        val_trajectories_np   = val_trajectories.detach().cpu().numpy()
        val_params_np         = val_params.detach().cpu().numpy()
        val_indices_np        = val_indices.detach().cpu().numpy()

        fname = f"{prefix}_nsrsd{nsr_sd:.3f}_tauc{tau_c:.3f}_rep{rep:02d}.npz"
        fpath = os.path.join(out_dir, fname)

        np.savez(
            fpath,
            train_trajectories=train_trajectories_np,
            train_params=train_params_np,
            train_indices=train_indices_np,
            val_trajectories=val_trajectories_np,
            val_params=val_params_np,
            val_indices=val_indices_np
        )

    

if __name__ == "__main__":
    main()

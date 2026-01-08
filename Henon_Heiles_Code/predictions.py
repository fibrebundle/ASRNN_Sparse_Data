import os
import glob
import torch
import numpy as np

from helper import (
    HamiltonianMLPNetwork,
    VerletIntegrator,
    generate_trajectories_batch,
    transform_trajectories_to_numpy,
    E_time_series,
    split_params,
    predict_trajectories,
    device,   
)

alpha1 = 0.4
alpha2 = 0.6
alpha1_ = 4
alpha2_ = 6
folder_path = "./Initial_Conditions"
file_name = f"{folder_path}/in_conds_params_{alpha1_}_{alpha2_}.npy"

in_conds = torch.from_numpy(np.load(file_name)).float()
n_ic = in_conds.shape[0]
params_np = np.tile([alpha1, alpha2], (n_ic, 1))
params = torch.from_numpy(params_np).float()


pred_base_dir = f"./Noise_10_Percent/tau_2.5/Predictions/Params_{alpha1_}_{alpha2_}"
os.makedirs(pred_base_dir, exist_ok=True)


model_paths = sorted(glob.glob("./Noise_10_Percent/tau_2.5/Models/*_model.pt"))

for mpath in model_paths:
    stem = os.path.splitext(os.path.basename(mpath))[0]  
    base_stem = stem[:-len("_model")] if stem.endswith("_model") else stem

    model_network = HamiltonianMLPNetwork(
        kin_hidden_dim=30, kin_n_hidden=3,
        pot_hidden_dim=30, pot_n_hidden=3,
        device=device
    )
    state = torch.load(mpath, map_location=device)
    model_network.load_state_dict(state)
    model_network.eval()
    integrator = VerletIntegrator(model_network, dt=0.1)

    p_preds, q_preds, K_preds, V_preds = predict_trajectories(
        model=model_network,
        integrator=integrator,
        in_conds=in_conds,
        params=params,
        T=500
    )

    qs_pred_np = q_preds.detach().cpu().numpy()
    ps_pred_np = p_preds.detach().cpu().numpy()
    Ks_pred_np = K_preds.detach().cpu().numpy()[:, :, 0]
    Vs_pred_np = V_preds.detach().cpu().numpy()[:, :, 0]
    Es_pred_np = Ks_pred_np + Vs_pred_np

    pred_trajectories = torch.cat([p_preds, q_preds], dim=2)  
    pred_trajectories_transformed = transform_trajectories_to_numpy(pred_trajectories)  
    

    ground_trajectories = generate_trajectories_batch(
        in_conds, params, T=500, dt=0.1, coarsening_factor=100
    )
    ground_trajectories_np = transform_trajectories_to_numpy(ground_trajectories)
    qps = ground_trajectories_np.transpose(2, 1, 0)
    ps_ground_np = qps[:, :, 2:]
    qs_ground_np = qps[:, :, :2]
    alphas, betas = split_params(params)
    
    Vs_pred_from_traj, Ks_pred_from_traj, Es_pred_from_traj = E_time_series(pred_trajectories_transformed, alphas, betas)
    Vs_pred_from_traj_np = np.transpose(Vs_pred_from_traj, axes=(1, 0))
    Ks_pred_from_traj_np = np.transpose(Ks_pred_from_traj, axes=(1, 0))
    Es_pred_from_traj_np = Ks_pred_from_traj_np + Vs_pred_from_traj_np
    Vs_ground, Ks_ground, Es_ground = E_time_series(ground_trajectories_np, alphas, betas)
    Vs_ground_np = np.transpose(Vs_ground, axes=(1, 0))
    Ks_ground_np = np.transpose(Ks_ground, axes=(1, 0))
    Es_ground_np = Ks_ground_np + Vs_ground_np

    offset_K = Ks_pred_from_traj_np[0, :] - Ks_pred_np[0, :]  
    offset_V = Vs_pred_from_traj_np[0, :] - Vs_pred_np[0, :]  
    offset_E = Es_pred_from_traj_np[0, :] - Es_pred_np[0, :]  
    Ks_pred_corrected_np = Ks_pred_np + offset_K[np.newaxis, :]  
    Vs_pred_corrected_np = Vs_pred_np + offset_V[np.newaxis, :]  
    Es_pred_corrected_np = Es_pred_np + offset_E[np.newaxis, :]  

    out_path = os.path.join(pred_base_dir, f"{base_stem}_predictions.npz")
    np.savez(
        out_path,
        params=params_np,
        ps_ground=ps_ground_np, qs_ground=qs_ground_np,
        ps_pred=ps_pred_np, qs_pred=qs_pred_np,
        Ks_ground=Ks_ground_np, Vs_ground=Vs_ground_np,
        Ks_pred=Ks_pred_np, Vs_pred=Vs_pred_np, Es_ground=Es_ground_np, 
        Es_pred=Es_pred_np,
        Ks_pred_from_traj=Ks_pred_from_traj_np, Vs_pred_from_traj=Vs_pred_from_traj_np,
        Es_pred_from_traj=Es_pred_from_traj_np,
        Ks_pred_corrected=Ks_pred_corrected_np, Vs_pred_corrected=Vs_pred_corrected_np,
        Es_pred_corrected=Es_pred_corrected_np
    )


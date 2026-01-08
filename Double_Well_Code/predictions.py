import os
import glob
import torch
import numpy as np

from helper import (
    HamiltonianMLPNetwork,
    VerletIntegrator,
    generate_trajectories_batch,
    predict_trajectories,
    K,
    V,
    device,  
)

alpha = 0.8
neg_alpha = -0.8
folder_path = "./Initial_Conditions"
file_name = f"{folder_path}/in_conds_param_{alpha}.npy"
#file_name = f"{folder_path}/in_conds_param_neg_{alpha}.npy"

in_conds = torch.from_numpy(np.load(file_name)).float()
n_ic = in_conds.shape[0]
params_np = np.tile([alpha], (n_ic, 1))
#params_np = np.tile([neg_alpha], (n_ic, 1))
params = torch.from_numpy(params_np).float()
alpha_val = float(params[0, 0].item())


pred_base_dir = f"./No_Noise/Predictions/Param_{alpha}"
#pred_base_dir = f"./Noise_10_Percent/tau_2.5/Predictions/Param_neg_{alpha}"
os.makedirs(pred_base_dir, exist_ok=True)


model_paths = sorted(glob.glob("./No_Noise/Models/*_model.pt"))

for mpath in model_paths:
    stem = os.path.splitext(os.path.basename(mpath))[0]   
    base_stem = stem[:-len("_model")] if stem.endswith("_model") else stem

    model_network = HamiltonianMLPNetwork(
        kin_hidden_dim=50, kin_n_hidden=2,
        pot_hidden_dim=50, pot_n_hidden=2,
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
        T=200
    )

    qs_pred_np = q_preds.detach().cpu().numpy()
    ps_pred_np = p_preds.detach().cpu().numpy()
    Ks_pred_np = K_preds.detach().cpu().numpy()[:, :, 0]
    Vs_pred_np = V_preds.detach().cpu().numpy()[:, :, 0]
    Es_pred_np = Ks_pred_np + Vs_pred_np

    ground_trajectories = generate_trajectories_batch(
        in_conds, params, T=200, dt=0.1, coarsening_factor=100
    )
    ps_ground = ground_trajectories[:, :, 0:1]
    qs_ground = ground_trajectories[:, :, 1:2]
    Ks_ground = K(ps_ground)
    Vs_ground = V(alpha_val, qs_ground)
    Es_ground = Ks_ground + Vs_ground
    ps_ground_np = ps_ground.detach().cpu().numpy()
    qs_ground_np = qs_ground.detach().cpu().numpy()
    Ks_ground_np = Ks_ground[:, :, 0].detach().cpu().numpy()
    Vs_ground_np = Vs_ground[:, :, 0].detach().cpu().numpy()
    Es_ground_np = Ks_ground_np + Vs_ground_np

    Vs_pred_from_traj_np = V(alpha_val, q_preds)[:, :, 0].detach().cpu().numpy()
    Ks_pred_from_traj_np = K(p_preds)[:, :, 0].detach().cpu().numpy()
    Es_pred_from_traj_np = Vs_pred_from_traj_np + Ks_pred_from_traj_np
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
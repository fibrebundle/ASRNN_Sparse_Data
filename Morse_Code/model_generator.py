import os
import glob
import numpy as np
import torch
from torch.optim import LBFGS

from helper import Hamiltonian_MLP_Network, VerletIntegrator, residuals


def main():
    data_glob       = "./Noise_10_Percent/tau_2.5/Training_Data/*.npz"
    out_dir_models  = "./Noise_10_Percent/tau_2.5/Models"
    out_dir_losses  = "./Noise_10_Percent/tau_2.5/Losses"
    kin_hidden_dim  = 50
    kin_n_hidden    = 2
    pot_hidden_dim  = 50
    pot_n_hidden    = 2
    T               = 15        
    steps           = 500       
    dt              = 0.1       
   
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dtype  = torch.float32

    os.makedirs(out_dir_models, exist_ok=True)
    os.makedirs(out_dir_losses, exist_ok=True)
    dataset_paths = sorted(glob.glob(data_glob))
    

    for i, fpath in enumerate(dataset_paths, start=1):
        stem = os.path.splitext(os.path.basename(fpath))[0]

        data = np.load(fpath)
        train_trajectories_np = data["train_trajectories"]
        train_params_np       = data["train_params"]
        train_indices_np      = data["train_indices"]
        val_trajectories_np   = data["val_trajectories"]
        val_params_np         = data["val_params"]
        val_indices_np        = data["val_indices"]

        train_trajectories = torch.from_numpy(train_trajectories_np).to(device=device, dtype=dtype)
        train_params       = torch.from_numpy(train_params_np).to(device=device, dtype=dtype)
        train_instants     = torch.from_numpy(train_indices_np).to(device=device)  
        val_trajectories   = torch.from_numpy(val_trajectories_np).to(device=device, dtype=dtype)
        val_params         = torch.from_numpy(val_params_np).to(device=device, dtype=dtype)
        val_instants       = torch.from_numpy(val_indices_np).to(device=device)

       
        model = Hamiltonian_MLP_Network(
            kin_hidden_dim=kin_hidden_dim, kin_n_hidden=kin_n_hidden,
            pot_hidden_dim=pot_hidden_dim, pot_n_hidden=pot_n_hidden,
            device=device
        ).to(device=device)
        integrator = VerletIntegrator(model=model, dt=dt)

        optimizer = LBFGS(
            model.parameters(),
            lr=1.0, history_size=10, line_search_fn="strong_wolfe",
            tolerance_grad=1e-32, tolerance_change=1e-32
        )

        training_losses   = []
        validation_losses = []

        def closure():
            optimizer.zero_grad()
            loss = residuals(train_trajectories, train_params, train_instants, T, integrator)
            loss.backward()
            return loss

        for step in range(steps):
            model.train()
            train_loss = float(optimizer.step(closure).item())

            model.eval()
            val_loss_tensor = residuals(val_trajectories, val_params, val_instants, T, integrator)
            val_loss = float(val_loss_tensor.detach().cpu())

            training_losses.append(train_loss)
            validation_losses.append(val_loss)


        model_out = os.path.join(out_dir_models, f"{stem}_model.pt")
        losses_out = os.path.join(out_dir_losses, f"{stem}_losses.npz")

        torch.save(model.state_dict(), model_out)
        np.savez(losses_out,
                 training_losses=np.asarray(training_losses, dtype=np.float64),
                 validation_losses=np.asarray(validation_losses, dtype=np.float64))


if __name__ == "__main__":
    main()

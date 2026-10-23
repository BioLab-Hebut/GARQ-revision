import torch
import numpy as np
from model import reconstruction_loss  


def init_gart_anchors(
    model,
    data_types,
    dataloader,
    device,
    init_method="Kmeans"  
):

    model.eval()  
    all_hiddens = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= 2:  
                break
            
            x_list = []
            for i in range(model.omics_num):
                x_list.append(data["x"][i].to(device, non_blocking=True))
            
     
            hiddens = model(x_list)
            if model.omics_num > 1:
                hidden = torch.cat(hiddens, dim=1)
            else:
                hidden = hiddens[0]
            
            all_hiddens.append(hidden.detach())
    
 
    all_hidden = torch.cat(all_hiddens, dim=0)
    model.init_quantizer(all_hidden, method=init_method)  
    model.train()  
    print(f"GARQ Initialization completed.")  


def train_one_epoch(
    model,
    data_types,
    dataloader,
    optimizer,
    epoch,
    device,
):
    model.train(True)
    optimizer.zero_grad()

    omics_num = model.omics_num

    loss_rec_epoch = [0 for _ in range(omics_num)]
    loss_rec_q_epoch = [0 for _ in range(omics_num)]
    loss_c_epoch = 0

    for data in dataloader:
        x_list = []
        sf_list = []
        raw_list = []
        for i in range(omics_num):
            x_list.append(data["x"][i].to(device, non_blocking=True))
            sf_list.append(data["sf"][i].to(device, non_blocking=True))
            raw_list.append(data["raw"][i].to(device, non_blocking=True))

        hiddens = model(x_list)
        hidden_q, loss_c = model.quantize(hiddens)

        means, disps, pis = model.decode(hiddens)           
        means_q, disps_q, pis_q = model.decode_q(hidden_q)   

        loss_rec_all = 0
        loss_rec_q_all = 0
        for i in range(omics_num):
            loss_rec = reconstruction_loss(
                means[i], disps[i], sf_list[i], raw_list[i], data_types[i], pi=pis[i]
            )
            loss_rec_q = reconstruction_loss(
                means_q[i], disps_q[i], sf_list[i], raw_list[i], data_types[i], pi=pis_q[i]
            )
            loss_rec_all += loss_rec
            loss_rec_q_all += loss_rec_q
            loss_rec_epoch[i] += loss_rec.item()
            loss_rec_q_epoch[i] += loss_rec_q.item()
        loss_c_epoch += loss_c.item()

        loss = loss_rec_all + loss_rec_q_all + loss_c

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print("Epoch %d," % (epoch + 1), end=" ")
        for i in range(omics_num):
            if i > 0:
                print(",", end=" ")
            print(
                data_types[i]
                + ": recon_loss=%.4f" % (loss_rec_epoch[i] / len(dataloader)),
                end=" ",
            )
            print(
                "recon_loss Q=%.4f" % (loss_rec_q_epoch[i] / len(dataloader)),
                end=" ",
            )
        print(", Anchers: loss=%.4f" % (loss_c / len(dataloader)))

    loss_rec_q_epoch = np.array(loss_rec_q_epoch).mean()  
    loss_c_epoch = loss_c_epoch / len(dataloader)

    return loss_rec_q_epoch, loss_c_epoch


def warm_one_epoch(
    model,
    data_types,
    dataloader,
    optimizer,
    epoch,
    device,
):
    model.train(True)
    optimizer.zero_grad()

    omics_num = model.omics_num

    loss_rec_epoch = [0 for _ in range(omics_num)] 

    for data in dataloader:
        x_list = []
        sf_list = []
        raw_list = []
        for i in range(omics_num):
            x_list.append(data["x"][i].to(device, non_blocking=True))
            sf_list.append(data["sf"][i].to(device, non_blocking=True))
            raw_list.append(data["raw"][i].to(device, non_blocking=True))

        hiddens = model(x_list) 
        means, disps,pis = model.decode(hiddens) 

        loss_rec_all = 0
        for i in range(omics_num):
            loss_rec = reconstruction_loss(
                means[i], disps[i], sf_list[i], raw_list[i], data_types[i], pi=pis[i]
            )
            loss_rec_all += loss_rec 
            loss_rec_epoch[i] += loss_rec.item() 

        loss = loss_rec_all 

        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print("Epoch %d," % (epoch + 1), end=" ")
        for i in range(omics_num):
            if i > 0:
                print(",", end=" ")
            print(
                data_types[i]
                + ": recon_loss=%.4f" % (loss_rec_epoch[i] / len(dataloader)),
                end=" ",
            )
        print("")
    loss_rec_epoch = np.array(loss_rec_epoch).mean() / len(dataloader)

    return loss_rec_epoch


@torch.no_grad()
def inference(model, data_types, data_loader, device):
    model.eval()

    omics_num = model.omics_num

    embeds = []
    ids = []
    delta_confs = []
    loss_rec_all = 0
    loss_rec_q_all = 0
    loss_c_all = 0

    for data in data_loader:
        x_list = []
        sf_list = []
        raw_list = []
        for i in range(omics_num):
            x_list.append(data["x"][i].to(device, non_blocking=True))
            sf_list.append(data["sf"][i].to(device, non_blocking=True))
            raw_list.append(data["raw"][i].to(device, non_blocking=True))

        with torch.no_grad():
            hiddens = model(x_list)
            id, delta_conf, loss_c = model.quantize(hiddens, return_assignment=True)
            loss_c_all += loss_c

            hidden_q, _ = model.quantize(hiddens)
            means, disps,pis = model.decode(hiddens)
            means_q, disps_q,pis_q = model.decode_q(hidden_q)

            for i in range(omics_num):
                loss_rec = reconstruction_loss(
                    means[i],
                    disps[i],
                    sf_list[i],
                    raw_list[i],
                    data_types[i],
                    pi=pis[i],
                    reduction="sum",
                )
                loss_rec_all += loss_rec
                loss_rec_q = reconstruction_loss(
                    means_q[i],
                    disps_q[i],
                    sf_list[i],
                    raw_list[i],
                    data_types[i],
                    pi=pis_q[i],
                    reduction="sum",
                )
                loss_rec_q_all += loss_rec_q

        if omics_num > 1:
            hidden = torch.cat(hiddens, dim=1)
        else:
            hidden = hiddens[0]

        embeds.append(hidden.detach().cpu().numpy())
        ids.append(id.detach().cpu().numpy())
        delta_confs.append(delta_conf.detach().cpu().numpy())

    embeds = np.concatenate(embeds, axis=0)
    ids = np.concatenate(ids, axis=0)
    delta_confs = np.concatenate(delta_confs, axis=0)

    rec_q_percent = (loss_rec_all / loss_rec_q_all).item()
    loss_c_all = loss_c_all.item() / (
        data_loader.dataset.__len__() * model.quantizer.entry_dim
    )

    return embeds, ids, delta_confs, rec_q_percent, loss_c_all

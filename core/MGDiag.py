import os
import time
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm.auto import tqdm
from core.ita import cal_task_affinity
from core.loss.UnsupervisedContrastiveLoss import UspConLoss
from core.loss.SupervisedContrastiveLoss import SupConLoss
from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.model.MainModel import MainModel

from helper.eval import *
from helper.early_stop import EarlyStopping
from helper.Result import Result
from config.exp_config import Config

class TVDiag(object):

    def __init__(self, config: Config, logger, log_dir: str):
        self.config = config
        self.logger = logger
        self.logger.info("Initialize runtime and output directories")
        os.makedirs(log_dir, exist_ok=True)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Currently using GPU {}".format(config.gpu_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
            self.device='cuda'
        else:
            logger.info("Currently using CPU (GPU is highly recommended)")
            self.device = 'cpu'

        self.result = Result()
        # alias tensorboard='python3 -m tensorboard.main'
        # tensorboard --logdir=logs --host=0.0.0.0 --port=30000
        self.writer = SummaryWriter(log_dir)
        self.logger.info("TensorBoard output directory: %s", self.writer.log_dir)
        self.logger.info("Experiment configuration:")
        self.printParams()

    def printParams(self):
        self.config.print_configs(self.logger)
    

    def train(self, train_data, aug_data):
        self.logger.info("[Training 1/4] Create model and optimizer")
        model = MainModel(self.config).to(self.device)
        awl = AutomaticWeightedLoss(4).to(self.device)
        optimizer_parameters = list(model.parameters())
        if self.config.dynamic_weight:
            optimizer_parameters += list(awl.parameters())
        opt = torch.optim.Adam(
            optimizer_parameters,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.logger.info("[Training 2/4] Initialize loss functions")
        supConLoss = SupConLoss(self.config.temperature, self.device).to(self.device)
        uspConLoss = UspConLoss(self.config.temperature, self.device).to(self.device)

        self.logger.debug(model)
        self.logger.info(
            "[Training 3/4] Start training: epochs=%d, batch_size=%d, samples=%d",
            self.config.epochs,
            self.config.batch_size,
            len(train_data),
        )
        
        # Overhead
        train_times = []

        # Inter-Task Affinity (RCL -> FTI, FTI -> RCL)
        Z_r2fs, Z_f2rs = [], []
        
        # early stop configuration
        earlyStop = EarlyStopping(patience=self.config.patience)
        epoch_bar = tqdm(
            range(self.config.epochs),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            ascii=True,
        )
        for epoch in epoch_bar:
            
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss, epoch_con_l, epoch_rcl_l, epoch_fti_l = 0, 0, 0, 0
            rcl_results = {
                **{f"HR@{k}": [] for k in range(1, 6)},
                **{f"RootRecall@{k}": [] for k in range(1, 6)},
                "MRR@3": [],
            }
            fti_results = {'pre':[], 'rec':[], 'f1':[]}

            # train_data and randomly sampled aug_data
            train_dl = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, collate_fn=self.collate)
            batch_bar = tqdm(
                train_dl,
                total=len(train_dl),
                desc=f"Epoch {epoch + 1}/{self.config.epochs}",
                unit="batch",
                dynamic_ncols=True,
                ascii=True,
                leave=False,
            )
            for batch_graphs, batch_labels in batch_bar:
                batch_graphs = batch_graphs
                instance_labels = batch_labels[:, 0]
                type_labels = batch_labels[:, 1]

                if self.config.aug_times > 0:
                    raw_graphs = dgl.unbatch(batch_graphs)
                    aug_graphs, aug_labels = map(list, zip(*random.sample(aug_data, len(raw_graphs))))
                    batch_graphs = dgl.batch(raw_graphs + aug_graphs)
                    instance_labels = torch.hstack((instance_labels, torch.tensor(aug_labels)[:,0].flatten()))
                    type_labels = torch.hstack((type_labels, torch.tensor(aug_labels)[:,1].flatten()))


                batch_graphs = batch_graphs.to(self.device)
                instance_labels = instance_labels.to(self.device)
                type_labels = type_labels.to(self.device)

                opt.zero_grad()

                fs, es, root_logit, type_logit = model(batch_graphs)

                '''
                    Task-oriented learning
                '''
                l_to, l_cm = torch.tensor(0.0, dtype=torch.float32, device=self.device), \
                    torch.tensor(0.0, dtype=torch.float32, device=self.device)
                modalities = self.config.modalities
                if self.config.TO:
                    if 'metric' in modalities:
                        l_to += supConLoss(fs['metric'], instance_labels)
                        # l_to += supConLoss(fs['metric'], type_labels)
                    if 'log' in modalities:
                        l_to += supConLoss(fs['log'], type_labels)
                    if 'trace' in modalities:
                        l_to += supConLoss(fs['trace'], instance_labels)

                '''
                    Cross-modal association
                '''
                if self.config.CM:
                    if len(modalities) >= 2 and 'metric' in modalities:
                        left_modalies = [modality for modality in modalities if modality != 'metric']
                        for modality in left_modalies:
                            l_cm += uspConLoss(fs['metric'], fs[modality])
                
                '''
                    Failure Diagnosis
                '''
                sigma = self.config.contrastive_loss_scale
                l_con = sigma * (l_to + l_cm)

                # RCL Loss
                l_rcl = self.cal_rcl_loss(root_logit, batch_graphs)
                # FTI Loss
                l_fti = F.cross_entropy(type_logit, type_labels)
                if self.config.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti, sigma*l_to, sigma*l_cm)
                else:
                    total_loss = l_con + l_rcl + l_fti

                self.logger.debug("con_loss: {:.3f}, RCA_loss: {:.3f}, TC_loss: {:.3f}"
                      .format(l_con, l_rcl, l_fti))

                
                # Calculate Inter-Task Affinity
                if epoch == 0:
                    Z_r2f, Z_f2r = cal_task_affinity(model=model, 
                                    optimizer=opt, 
                                    batch_graphs=batch_graphs,
                                    type_labels=type_labels,
                                    device=self.device)
                    Z_r2fs.append(Z_r2f)
                    Z_f2rs.append(Z_f2r)

                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.detach().item()
                epoch_con_l += l_con.detach().item()
                epoch_rcl_l += l_rcl.detach().item()
                epoch_fti_l += l_fti.detach().item()

                rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                fti_res = FTI_eval(type_logit, type_labels)
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]

                n_iter += 1
                batch_bar.set_postfix(
                    loss=f"{epoch_loss / n_iter:.4f}",
                    rca=f"{epoch_rcl_l / n_iter:.4f}",
                    fti=f"{epoch_fti_l / n_iter:.4f}",
                )
                
            mean_epoch_loss = epoch_loss / n_iter
            mean_con_loss = epoch_con_l / n_iter
            mean_rcl_loss = epoch_rcl_l / n_iter
            mean_fti_loss = epoch_fti_l / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            train_times.append(time_per_epoch)
            epoch_bar.set_postfix(
                loss=f"{mean_epoch_loss:.4f}",
                rca=f"{mean_rcl_loss:.4f}",
                fti=f"{mean_fti_loss:.4f}",
            )
            self.logger.info(
                "Epoch %d/%d done: loss=%.4f, contrastive=%.4f, RCA=%.4f, FTI=%.4f, time=%.2fs",
                epoch + 1,
                self.config.epochs,
                mean_epoch_loss,
                mean_con_loss,
                mean_rcl_loss,
                mean_fti_loss,
                time_per_epoch,
            )

            for k, v in rcl_results.items():
                rcl_results[k] = np.mean(v)
            for k, v in fti_results.items():
                fti_results[k] = np.mean(v)
            self.writer.add_scalar('loss/mean total loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean con loss', mean_con_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean RCL loss', mean_rcl_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean FTI loss', mean_fti_loss, global_step=epoch)
            self.writer.add_scalar('train/HR@1', rcl_results['HR@1'], global_step=epoch)
            self.writer.add_scalar('train/HR@3', rcl_results['HR@3'], global_step=epoch)
            self.writer.add_scalar('train/HR@5', rcl_results['HR@5'], global_step=epoch)
            self.writer.add_scalar('train/avg@3', np.mean([rcl_results['HR@1'], rcl_results['HR@2'], rcl_results['HR@3']]), global_step=epoch)
            self.writer.add_scalar('train/MRR@3', rcl_results['MRR@3'], global_step=epoch)
            self.writer.add_scalar('train/precision', fti_results['pre'], global_step=epoch)
            self.writer.add_scalar('train/recall', fti_results['rec'], global_step=epoch)
            self.writer.add_scalar('train/f1-score', fti_results['f1'], global_step=epoch)

            stop = earlyStop.should_stop(mean_epoch_loss, epoch)
            if stop:
                self.logger.info(
                    "Early stop at epoch %d due to %d epochs without improvement",
                    epoch + 1,
                    self.config.patience,
                )
                break
            # losses.append(mean_epoch_loss)
            # if len(losses) > 10 and abs(losses[-10] - losses[-1]) < self.config.patience:
            #     break

            
        epoch_bar.close()
        self.logger.info("[Training 4/4] Save checkpoint")
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        torch.save(state, os.path.join(self.writer.log_dir, 'TVDiag.pt'))
        self.writer.flush()

        self.result.set_train_efficiency(train_times)
        self.logger.info("Training finished after %d epochs", epoch + 1)
        # calculate the training time for raw data
        self.logger.debug(f"The training time is {self.result.total_train_time}[s]")
        self.logger.debug(f"The training time per epoch is {self.result.avg_train_time}[s]")
        self.logger.debug(f"The affinity of RCL -> FTI is {np.mean(Z_r2f)}")
        self.logger.debug(f"The affinity of FTI -> RCL is {np.mean(Z_f2r)}")
        self.logger.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


    def evaluate(self, test_data, model=None):
        if model is None:
            checkpoint_path = os.path.join(self.writer.log_dir, 'TVDiag.pt')
            self.logger.info("[Evaluation 1/3] Load checkpoint: %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model = MainModel(self.config).to(self.device)
            model.load_state_dict(checkpoint['model'])
       
        self.logger.info("[Evaluation 2/3] Run inference on %d samples", len(test_data))
        model.eval()
        root_logits, type_logits = [], []
        roots, types = [], []
        predictions = []
        inference_times = []
        num_node_list = []
        test_bar = tqdm(
            test_data,
            total=len(test_data),
            desc="Evaluation",
            unit="sample",
            dynamic_ncols=True,
            ascii=True,
        )
        for graph, labels, metadata in test_bar:
            graph = graph.to(self.device)
            failure_type = labels[1]
            roots.append(graph.ndata['root'])
            types.append(failure_type)
            num_node_list.append(graph.num_nodes())
        
            start_time = time.time()
            with torch.no_grad():
                _, _, root_logit, type_logit = model(graph)
                root_logits.append(root_logit.flatten())
                type_logits.append(type_logit.flatten())
                root_index = int(root_logit.flatten().argmax().item())
                type_index = int(type_logit.flatten().argmax().item())
                ranking = torch.argsort(root_logit.flatten(), descending=True).tolist()
                top_k = ranking[: min(5, len(ranking))]
                predictions.append({
                    "index": metadata["index"],
                    "predicted_root": metadata["nodes"][root_index],
                    "predicted_root_type": metadata["node_types"][root_index],
                    "predicted_failure_type": self.config.failure_types[type_index],
                    "top5_roots": ";".join(metadata["nodes"][index] for index in top_k),
                    "top5_root_types": ";".join(metadata["node_types"][index] for index in top_k),
                    "true_roots": ";".join(metadata["roots"]),
                    "true_root_types": ";".join(metadata["root_types"]),
                    "true_failure_type": metadata["failure_type"],
                })
            end_time = time.time()
            inference_times.append(end_time - start_time)
            test_bar.set_postfix(
                nodes=graph.num_nodes(),
                avg_ms=f"{np.mean(inference_times) * 1000:.2f}",
            )

        self.logger.info("[Evaluation 3/3] Aggregate predictions and calculate metrics")
        root_logits = torch.hstack(root_logits).cpu()
        type_logits = torch.vstack(type_logits).cpu()
        roots = torch.hstack(roots)
        types = torch.tensor(types)

        rcl_res = RCA_eval(root_logits, num_node_list, roots)
        fti_res = FTI_eval(type_logits, types)
        self.result.set_performance(rcl_res, fti_res)
        self.result.set_inference_efficiency(inference_times)
        self.result.set_predictions(predictions)


        avg_3 = np.mean([rcl_res['HR@1'], rcl_res['HR@2'], rcl_res['HR@3']])

        self.logger.info(
            "[Root localization] HR@1: %.3f, HR@3: %.3f, HR@5: %.3f, "
            "RootRecall@3: %.3f, MRR@3: %.3f",
            rcl_res['HR@1'], rcl_res['HR@3'], rcl_res['HR@5'],
            rcl_res['RootRecall@3'], rcl_res['MRR@3'],
        )
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}"\
            .format(fti_res['pre'], fti_res['rec'], fti_res['f1']))
        self.logger.info(f"The average test time is {np.mean(inference_times)}[s]")
        self.logger.info(f"The total test time is {np.sum(inference_times)}[s]")

        return self.result

    
    def cal_rcl_loss(self, root_logit, batch_graphs):        
        num_nodes_list = batch_graphs.batch_num_nodes()
        losses = []
        
        start_idx = 0
        for num_nodes in num_nodes_list:
            end_idx = start_idx + int(num_nodes)
            node_logits = root_logit[start_idx:end_idx].flatten()
            root_mask = batch_graphs.ndata["root"][start_idx:end_idx].bool()
            if not torch.any(root_mask):
                raise ValueError("graph has no positive root label")
            # Negative log probability mass assigned to any valid root. This
            # reduces to ordinary softmax cross entropy for a single root.
            loss = torch.logsumexp(node_logits, dim=0) - torch.logsumexp(
                node_logits[root_mask], dim=0
            )
            losses.append(loss)
            start_idx = end_idx
        return torch.stack(losses).mean()

    def collate(self, samples):
        graphs, labels, _metadata = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

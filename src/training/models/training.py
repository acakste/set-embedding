import logging, torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import warnings


from src.training.models.model_loader import ModelWrapper
from src.training.distortion.dist_metrics import wasserstein
from src.training.div.utils import select_function
from src.training.distortion.min_distortion import select_distortion_function
from src.training.distortion.train_to_reference import select_sample_function

# Different training objectives
from src.training.distortion.min_distortion import distortion_loss
from src.training.distortion.min_distortion import DistortionCriterion
# from src.training.models.models import neg_ll_GMM, neg_ll_GMM_2
from src.evaluation.GMM_likelihood_evaluation import neg_ll_GMM, neg_ll_GMM_2


from geomloss import SamplesLoss

# Logging
logger = logging.getLogger(__name__)



class LossCriterion:
    def __init__(self, criterion, type="normal", sample_func=None, reference_set=None, precomputed_dist=None, label_id_mapping=None, sample_loader=None, sample_size=1, max_num_tracks=100):
        """
        :param type: if "normal", a standard type of criterion: criterion(input, target)
                    else, distortion-based and pairwise comparisons.
        :param criterion: MSELoss, etc
        :param sample_func: if type is not "normal", use the sample_func to extract the reference set.
        """
        self.type = type
        self.criterion = criterion
        self.sample_func = sample_func
        self.reference_set = reference_set
        self.precomputed_dist = precomputed_dist
        self.sample_loader = sample_loader
        self.label_id_mapping = label_id_mapping
        self.sample_size = sample_size
        self.max_num_tracks = max_num_tracks
        self.input_distance_func = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    def __call__(self, model, x, label=None):
        """

        :param model: The pytorch model to feed the input through.
        :param x: A tensor of shape [B, L, d]. B is the batch size, L is the set size, and d is the dimensionality of the data.
        :param label: Only used in MinDistortion type of criterion. This also requires that
        :return:
        """

        if self.type == "normal":
            if label != None:
                raise Warning("Label is not used in this type of criterion.")
            return self.criterion(x, model(x))
        elif self.type == "MinDistortion":
            B, L, d = x.shape
            assert B == 1, "Batch size must be 1 for MinDistortion type of criterion."

            # Embed the input
            x_embedded = model(x)
            _, d_embed = x_embedded.shape

            input_dists = torch.zeros(size=(self.sample_size,))
            sample_set_embedded = torch.zeros(size=(self.sample_size, d_embed))
            for i in range(self.sample_size):
                sample_x, sample_label = next(iter(self.sample_loader))
                sample_set_embedded[i] = model(sample_x)

                # Calculate the input distances.
                if self.precomputed_dist != None:
                    # Find the id of the input and sample to be used in the precomputed_dist matrix.
                    x_id = self.label_id_mapping["label2id"][label[0]]
                    sample_id = self.label_id_mapping["label2id"][sample_label[0]]
                    input_dists[i] = self.precomputed_dist[x_id, sample_id].detach()
                else:
                    # Calculate the input distances on the fly.
                    _, L_x, _ = x.shape
                    _, L_sample_x, _ = sample_x.shape

                    indices_A = torch.randperm(L_x)[:self.max_num_tracks]
                    indices_B = torch.randperm(L_sample_x)[:self.max_num_tracks]

                    A = x.squeeze(0)[indices_A]
                    B = sample_x.squeeze(0)[indices_B]

                    input_dists[i] = self.input_distance_func(A.detach(), B.detach())**(1/2)

            # Calculate the embedding distances
            embedding_dists = torch.sum((sample_set_embedded - x_embedded)**2, dim=1)**(1/2)
            warnings.warn("Should we scale the input and/or the embedding distance by the dimension?")

            # Apply criterion and calculate loss
            return self.criterion(embedding_dists, input_dists)

        elif self.type == "AmortizedClustering":
            x_embed, pi, mu, sigma = model(x)
            return self.criterion(x, (pi, mu, sigma))
        else:
            raise ValueError("This type is not implemented")


class ModelTrainer:
    def __init__(self, model: ModelWrapper, data, exp_config, precomputed_dist=None, label_id_mapping=None):
        self.model = model
        self.data = data
        self.precomputed_dist = precomputed_dist
        self.label_id_mapping = label_id_mapping
        self.exp_config = exp_config

        if (precomputed_dist == None and label_id_mapping != None) or (precomputed_dist != None and label_id_mapping == None):
            raise ValueError("Both precomputed_dist and label_id_mapping must be None or not None.")

        if self.model.info["model_type"] == "XGBoost":
            self.train = self.train_xgboost
        elif self.model.info["model_type"] == "ParamEstimator":
            self.train = self.train_paramestimator
        else:
            self.train = self.train_loop
        """ Maybe nice to have the train_test as a standard loop, and then
            input criterion, loss, comparison method based on parameters."""

    def train_loop(self):
        """Set up criterion, loss, etc here"""
        model = self.model.model
        n_epochs = self.exp_config["n_epochs"]

        """ For backwards compatability. """
        try:
            lr = self.exp_config["lr"]
        except KeyError:
            lr = 1e-3
        try:
            weight_decay = self.exp_config["weight_decay"]
        except KeyError:
            weight_decay = 1e-5

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        dataloader = DataLoader(self.data, batch_size=self.exp_config["batch_size"], shuffle=True)

        if self.exp_config["objective"] == "MinDistortion":
            """ Training for minimum distortion. """
            add_noise = self.exp_config["MinDistortion"]["add_noise"]
            if add_noise == True:
                warnings.warn("We are adding randn fixed scaled noise (0.05 factor) to the input data.")

            # For DistortionCriterion & LossCriterion class
            distortion_func = select_distortion_function(self.exp_config["MinDistortion"]["distortion_func"])
            reduction = self.exp_config["MinDistortion"]["reduction"]
            # sample_func = select_sample_function(self.exp_config["MinDistortion"]["sample_func"]) # Not used, maybe if we want something other than random sampling.
            criterion = DistortionCriterion(distortion_func=distortion_func, reduction=reduction)

            if self.exp_config["MinDistortion"]["reference_set"] == "training_set":
                random_sampler = RandomSampler(data_source=dataloader.dataset)
                comparison_loader = DataLoader(dataloader.dataset, batch_size=1, sampler=random_sampler)
            else:
                raise AttributeError("Implement such that we can train to a fixed reference set.")

            """ To handle backwards compatability without having max_num_tracks set."""
            try:
                max_num_tracks = self.exp_config["MinDistortion"]["max_num_tracks"]
            except KeyError:
                max_num_tracks = 1000

            # Define the loss criterion to be used in the training loop.
            calculate_loss = LossCriterion(criterion=criterion,
                                           type="MinDistortion",
                                           sample_loader=comparison_loader,
                                           precomputed_dist=self.precomputed_dist,
                                           label_id_mapping=self.label_id_mapping,
                                           sample_size=self.exp_config["MinDistortion"]["sample_size"],
                                           max_num_tracks=max_num_tracks)

        elif self.exp_config["objective"] == "AmortizedClustering":
            add_noise = False
            calculate_loss = LossCriterion(criterion=neg_ll_GMM, type="AmortizedClustering")
        else:
            raise AttributeError("This objective implemented yet.")

        # Set the device to use.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using Device: {device}")

        losses = []
        for epoch in range(n_epochs):
            avg_loss_list = []
            for i, (x, label) in enumerate(dataloader):
                B, L, d = x.shape

                if add_noise:
                    noise = 0.05*torch.randn(size=(B,L,d))
                    x_input = x + noise
                else:
                    x_input = x

                # Move input to device, which is either cpu or cuda.
                x_input = x_input.to(device)

                # Calculate the loss
                loss = calculate_loss(model, x_input, label=label)

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                avg_loss_list.append(loss.item())

            avg_epoch_loss = np.mean(avg_loss_list)
            losses.append(avg_epoch_loss)

            # Write to logger
            logger.info(f"Epoch: {epoch}. Avg Loss: {avg_epoch_loss}")
            print(f"Epoch: {epoch}. Avg Loss: {avg_epoch_loss}")
        return losses

    def train_paramestimator(self):
        """Set up criterion, loss, etc here"""
        model = self.model.model
        n_epochs = self.exp_config["n_epochs"]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        dataloader = DataLoader(self.data, batch_size=self.exp_config["batch_size"], shuffle=True)
        add_noise = False

        """ Set the device to use. """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Using Device: {device}")

        losses = []
        for epoch in range(n_epochs):
            avg_loss_list = []
            for i, (x, label) in enumerate(dataloader):
                B, L, d = x.shape

                if add_noise:
                    noise = 0.05*torch.randn(size=(B,L,d))
                    x_input = x + noise
                else:
                    x_input = x

                # Move input to device, which is either cpu or cuda.
                x_input = x_input.to(device)

                x_embed, pi, mu, sigma = model(x_input)
                loss = neg_ll_GMM(x, (pi, mu, sigma), verbose=False)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                avg_loss_list.append(loss.item())

            avg_epoch_loss = np.mean(avg_loss_list)
            losses.append(avg_epoch_loss)

            """Add to logger"""
            logger.info(f"Epoch: {epoch}. Avg Loss: {avg_epoch_loss}")
            print(f"Epoch: {epoch}. Avg Loss: {avg_epoch_loss}")
        return losses
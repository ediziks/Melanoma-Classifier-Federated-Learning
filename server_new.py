
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path

from grpc import server

import flwr as fl
import tensorflow as tf
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
# from typing import Callable
import numpy as np
from datetime import datetime
import os
from tensorflow.keras import layers as L
# from tensorflow.keras.applications.efficientnet import EfficientNetB2 as efn
import efficientnet.tfkeras as efn


def load_model():
    IMAGE_SIZE = [384, 384]

    model = tf.keras.Sequential([
        efn.EfficientNetB2(
            input_shape=(*IMAGE_SIZE, 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(1024, activation = 'relu'), 
        L.Dropout(0.3), 
        L.Dense(512, activation= 'relu'), 
        L.Dropout(0.2), 
        L.Dense(256, activation='relu'), 
        L.Dropout(0.2), 
        L.Dense(128, activation='relu'), 
        L.Dropout(0.1), 
        L.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='Adam',
        loss = 'binary_crossentropy',
        metrics=['binary_crossentropy', 'accuracy'],
    )
    
    return model

model = load_model()
model.load_weights('./melamodel/melamodel_weights072.h5')


class SaveModelStrategy(fl.server.strategy.FedAvg):
    # def initialize_parameters(
    #     self, client_manager: ClientManager
    # ) -> Optional[Parameters]:
    #     """Initialize global model parameters."""
    #     # initial_parameters = self.initial_parameters
    #     # self.initial_parameters = None  # Don't keep initial parameters in memory
    #     initial_parameters = model.get_weights()
    #     return initial_parameters

    # def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    #     """Return a function which returns training configurations."""

    #     def fit_config(server_round: int) -> Dict[str, str]:
    #         """Return a configuration with static batch size and (local) epochs."""
    #         config = {
    #             "learning_rate": str(0.00001),
    #             "batch_size": str(8),
    #         }
    #         return config

    #     return fit_config

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[fl.common.Weights]:
        if not results:
            return None

        # model = load_model()
        # Weight accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")
        
        # only 2 decimal places on accuracy_aggregated
        accuracy_agg2 = round(accuracy_aggregated, 2) 
        aggregated_weights = super().aggregate_fit(server_round, results, failures)   
        aggregated_params, _ = aggregated_weights
        if aggregated_params is not None:
            aggregated_weights_h : List[np.ndarray] = fl.common.parameters_to_weights(aggregated_params)
            # modell = tf.keras.models.clone_model(model)
            model.set_weights(aggregated_weights_h)
            print(f'Federated Learning session completed! The accuracy of the aggregated model is {accuracy_agg2}')
            print(f"Saving round {server_round} model weights...")
            date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            model.save_weights(f"./workspace/clientResults/round-{server_round}-weights-{date}.h5")

        return aggregated_weights


# Create strategy and run server
strategy = SaveModelStrategy(
    # fraction_fit=0.01,
    initial_parameters=fl.common.weights_to_parameters(model.get_weights())
)
fl.server.start_server(strategy=strategy, config={"num_rounds": 1})

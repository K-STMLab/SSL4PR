import torch
from torch import nn
from transformers import AutoModel

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPoolingLayer(nn.Module):
    """
    This layer implement attention pooling.
    Given a sequence of vectors (bs, seq_len, embed_dim), it:
    1. Applies a linear layer to each vector (bs, seq_len, 1)
    2. Applies a softmax to each vector (bs, seq_len, 1)
    3. Applies a weighted sum to the sequence (bs, embed_dim)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: The input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            The output tensor of shape (batch_size, embed_dim).
        """
        # Linear layer (bs, seq_len, embed_dim) -> (bs, seq_len, 1)
        weights = self.linear(x)
        # Softmax (bs, seq_len, 1) -> (bs, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        # Weighted sum (bs, seq_len, 1) * (bs, seq_len, embed_dim) -> (bs, embed_dim)
        x = torch.sum(weights * x, dim=1)
        return x

class SSLClassificationModel(nn.Module):
    def __init__(
        self, 
        config: Optional[dict] = None,
    ):

        super(SSLClassificationModel, self).__init__()
        if config is None:
            raise ValueError("model_config must be provided to instantiate the classification model")
        
        self.config = config
        
        self.model_name_or_path = self.config.model.model_name_or_path
        self.num_classes = self.config.model.num_classes
        self.classifier_type = self.config.model.classifier_type
        self.num_layers = self.config.model.classifier_num_layers
        self.hidden_size = self.config.model.classifier_hidden_size
        self.dropout = self.config.model.dropout

        self.pooling_embedding_dim = 0
        self.global_embedding_dim = 0

        if config is None:
            raise ValueError("model_config must be provided to instantiate the classification model")

        if self.config.data.ssl:
            self.ssl = True
            self.set_ssl_model(config)
            self.pooling_embedding_dim += self.ssl_model.config.hidden_size
            self.global_embedding_dim += self.ssl_model.config.hidden_size
        else: self.ssl = False

        if self.config.data.magnitude:
            self.use_magnitudes = True
            if self.config.model.frame_fusion:
                self.pooling_embedding_dim += self.config.data.stft_params.spec_dim
            else:
                self.pooling_embedding_dim = self.config.data.stft_params.spec_dim
            self.global_embedding_dim += self.config.data.stft_params.spec_dim
        else: self.use_magnitudes = False

        # if config["data"]["articulation"]:
        #     self.articulation = True
        #     self.global_embedding_dim += 488
        # else: self.articulation = False
        # *** disable articulation features for now ***
        self.articulation = False

        print(f"Pooling embedding dim: {self.pooling_embedding_dim}")
        self.classifier, self.pooling_layer = self._get_classification_head(
            self.pooling_embedding_dim,
            self.global_embedding_dim,
        )

        self.init_weights()

    def init_weights(self):
        # initialize weights of classifier
        for name, param in self.classifier.named_parameters():
            if "weight" in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        
        # initialize weights of pooling layer
        for name, param in self.pooling_layer.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        return

    def set_ssl_model(self, config):
        if "whisper" in self.config.model.model_name_or_path:
            self.ssl_model = AutoModel.from_pretrained(self.model_name_or_path, output_hidden_states=self.config.model.use_all_layers)
            # take only the encoder part of the model
            self.ssl_model = self.ssl_model.encoder
            self.is_whisper = True
        else:
            self.ssl_model = AutoModel.from_pretrained(self.model_name_or_path, output_hidden_states=self.config.model.use_all_layers)
            self.is_whisper = False
            
        print(f"SSL model is whisper: {self.is_whisper}")
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.ssl_model.parameters() if p.requires_grad) / 1e6:.2f}M')
        
        if self.config.model.increase_resolution_cnn:
            # change last CNN layer setting stride to 1 instead of 2 (align time resolution)
            self.ssl_model.feature_extractor.conv_layers[6].conv.stride = (1,)
        
        if self.config.model.freeze_ssl:
            # freeze SSL model
            for param in self.ssl_model.parameters():
                param.requires_grad = False

        if self.config.model.use_all_layers:
            self.layer_weights = nn.Parameter(torch.ones(self.ssl_model.config.num_hidden_layers + 1))
            self.layer_weights.requires_grad = True
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.ssl_model.config.hidden_size) for _ in range(self.ssl_model.config.num_hidden_layers + 1)])
            self.layer_norms.requires_grad = True
            self.softmax = nn.Softmax(dim=-1)

        return
            
    def _get_classification_head(self, pooling_input_dim, global_input_dim):
        
        classifier = nn.Sequential()
        pooling_layer = nn.Sequential()
        
        # first layer
        if self.classifier_type == "average_pooling":
            # no additional layers for pooling
            pass
        elif self.classifier_type == "attention_pooling":
            pooling_layer.add_module(
                "attention_pooling_head", 
                AttentionPoolingLayer(pooling_input_dim)
            )

        # additional layers
        if self.config.model.classifier_head_type == "linear":
            for layer in range(self.num_layers):
                if layer == 0:
                    input_size = global_input_dim
                else:
                    input_size = self.hidden_size

                classifier.add_module(
                    f"layer_{layer}",
                    nn.Linear(input_size, self.hidden_size)
                )
                classifier.add_module(
                    f"layer_{layer}_activation",
                    nn.ReLU()
                )
                classifier.add_module(
                    f"layer_{layer}_dropout",
                    nn.Dropout(self.dropout)
                )
        elif self.config.model.classifier_head_type == "transformer":
            # input linear layer
            classifier.add_module(
                "input_layer",
                nn.Linear(global_input_dim, self.hidden_size)
            )
            
            # transformer layer
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.config.model.transformer_nhead,
                dim_feedforward=self.config.model.transformer_dim_feedforward,
                dropout=self.dropout,
                activation="relu",
                batch_first=True,
            )
            classifier.add_module(
                "transformer_layer",
                nn.TransformerEncoder(
                    encoder_layer=layer,
                    num_layers=self.num_layers,
                )
            )

        if self.num_classes == 2:
            # binary classification
            classifier.add_module(
                "final_layer",
                nn.Linear(self.hidden_size, 1)
            )
            classifier.add_module(
                "final_layer_activation",
                nn.Sigmoid()
            )
        else:
            # multi-class classification
            classifier.add_module(
                "final_layer",
                nn.Linear(self.hidden_size, self.num_classes)
            )
            # no softmax for cross-entropy loss
            # classifier.add_module(
            #     "final_layer_activation",
            #     nn.Softmax(dim=-1)
            # )

        return classifier, pooling_layer

    def get_ssl_features(self, input_values, **kwargs):

        if self.config.model.use_all_layers:
            if self.is_whisper:
                # forward pass through whisper model
                ssl_hidden_states = self.ssl_model(
                    input_features=input_values,
                    return_dict=True,
                ).hidden_states
            else:
                ssl_hidden_states = self.ssl_model(
                    input_values=input_values,
                    return_dict=True,
                ).hidden_states

            ssl_hidden_state = torch.zeros_like(ssl_hidden_states[-1])
            weights = self.softmax(self.layer_weights)
            for i in range(self.ssl_model.config.num_hidden_layers + 1):
                ssl_hidden_state += weights[i] * self.layer_norms[i](ssl_hidden_states[i])

        else:
            ssl_hidden_state = self.ssl_model(
                input_values=input_values,
                return_dict=True,
            ).last_hidden_state

        return ssl_hidden_state
    
    def _combine_features(self, ssl_features, magnitudes):
        # combine SSL and STFT features
        # print(f"SSL features shape: {ssl_features.shape}")
        # print(f"Magnitudes shape: {magnitudes.shape}")
        combined_features = torch.cat([ssl_features, magnitudes], dim=-1)
        return combined_features

    def forward(
        self,
        batch,
        **kwargs
    ):

        features = None

        if self.ssl:
            # forward pass through SSL model
            if self.is_whisper:
                ssl_input = batch["input_features"] 
            else:
                ssl_input = batch["input_values"]
            features = self.get_ssl_features(ssl_input)
        
        if self.use_magnitudes:
            magnitudes = batch["magnitudes"]
            if self.config.model.frame_fusion:
                features = self._combine_features(features, magnitudes)
            else:
                features = magnitudes
        
        # Do we want to use layer norm? TODO: add this to config
        
        if features is not None:
            if self.classifier_type == "average_pooling":
                # average pooling
                features = torch.mean(features, dim=1)
            else:
                # attention pooling
                features = self.pooling_layer(features)

        if self.articulation:
            print("Articulation features is NOT supported.")
            # articulation_features = batch["articulation"]
            # # concatenate articulation features with pooled output
            # if features is None:
            #     features = articulation_features
            # else:
            #     features = torch.cat([features, articulation_features], dim=-1)

        output = self.classifier(features)

        return output
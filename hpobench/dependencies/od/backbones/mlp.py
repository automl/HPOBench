import torch
from torch import nn

from hpobench.dependencies.od.utils.activations import ACTIVATIONS


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batch_normalization=False, dropout_rate=0.0):
        super().__init__()

        self.batch_normalization = batch_normalization
        self.linear_layer = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.batch_normalization_layer = nn.BatchNorm1d(out_channels)
        self.activation_layer = activation
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1

        z = self.linear_layer(x)

        if self.batch_normalization:
            # Batch normalization causes some troubles if it comes
            # to validation sanity run since mean/variance
            # are not initialized to that moment
            try:
                z = self.batch_normalization_layer(z)
            except:  # noqa E722
                pass

        z = self.activation_layer(z)
        z = self.dropout_layer(z)

        return z


class MLP(nn.Module):
    def __init__(self, num_features, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_features = num_features
        self._build_backbone()

    def _get_activation(self):
        if self.config["activation"] == "swish" or self.config["activation"] == "swish-1":
            train_beta = False
            if self.config["activation"] == "swish":
                train_beta = True

            return ACTIVATIONS["swish"](train_beta=train_beta)
        else:
            return ACTIVATIONS[self.config["activation"]]()

    def _build_backbone(self):
        features = self.num_features
        activation = self._get_activation()
        latent_dim = self.config["num_latent_units"]

        encoder_features = [features]
        for i in range(1, self.config["num_layers"]+1):
            encoder_features += [self.config[f"num_units_layer_{i}"]]

        decoder_features = [latent_dim] + encoder_features[::-1]

        features = encoder_features + decoder_features
        in_features = features.copy()  # We need different in_features if we use skip connections

        if self.config["skip_connection"]:
            # If skip connection
            # Usually we'd have the following:
            # 768 -> 128 -> 64 -> 8    -> 64     -> 128     -> 768
            # But since we merge the layers we get
            # 768 -> 128 -> 64 -> 8+64 -> 64+128 -> 128+768 -> 768
            decoder_index = int(len(features) / 2)
            encoder_index = decoder_index - 1
            for i in range(decoder_index, len(features)-1):
                in_features[i] = features[decoder_index] + features[encoder_index]
                encoder_index -= 1
                decoder_index += 1

            decoder_in_features = in_features[int(len(features) / 2):]
        else:
            decoder_in_features = in_features[int(len(features) / 2):]

        # Build encoder
        self.encoder_blocks = []
        for i in range(len(encoder_features)-1):
            self.encoder_blocks += [
                Block(
                    encoder_features[i],
                    encoder_features[i+1],
                    activation,
                    batch_normalization=self.config["batch_normalization"],
                    dropout_rate=0.0 if not self.config["dropout"] else self.config["dropout_rate"]
                )
            ]

        # Build decoder
        self.decoder_blocks = []
        for i in range(len(decoder_features)-2):
            self.decoder_blocks += [
                Block(
                    decoder_in_features[i],
                    decoder_features[i+1],
                    activation,
                    batch_normalization=self.config["batch_normalization"],
                    dropout_rate=0.0 if not self.config["dropout"] else self.config["dropout_rate"]
                )
            ]

        self.latent_dim = latent_dim
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.decoder_in_features = decoder_in_features

        # Make sure the parameters are within the model
        self.encoder_blocks = nn.Sequential(*self.encoder_blocks)
        self.decoder_blocks = nn.Sequential(*self.decoder_blocks)

    def encode(self, x):
        encoder_outputs = []

        output = x
        encoder_outputs += [output]

        # Processing encoder
        for block in self.encoder_blocks:
            output = block(output)
            encoder_outputs += [output]

        return encoder_outputs

    def decode(self, z, encoder_outputs=None):
        # Use encoder outputs only if skip connection is used
        if not self.config["skip_connection"]:
            encoder_outputs = None

        if encoder_outputs is not None:
            encoder_outputs = encoder_outputs[::-1]

        # Processing decoder
        output = z
        for i, block in enumerate(self.decoder_blocks):
            if encoder_outputs is not None:
                output = block(output, encoder_outputs[i])
            else:
                output = block(output)

        # concat if skip connection available
        if encoder_outputs is not None:
            output = torch.cat([output, encoder_outputs[-1]], dim=1)

        return output

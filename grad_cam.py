import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function


class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model.module._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.feature_module = feature_module
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input.cuda())
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1]
        B, C, Tg, _, _ = grads_val.size()
        weights = torch.mean(grads_val.view(B, C, Tg, -1), dim=3)
        weights = weights.view(B, C, Tg, 1, 1)
        activations = features[-1]
        localization_map = torch.sum(
            weights * activations, dim=1, keepdim=True)
        localization_map = F.relu(localization_map)
        localization_map = F.interpolate(
            localization_map,
            size=(64, 112, 112),
            mode="trilinear",
            align_corners=False,
        )
        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                   0
            ],
            torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                0
            ],
        )
        localization_map_min = torch.reshape(
            localization_map_min, shape=(B, 1, 1, 1, 1)
        )
        localization_map_max = torch.reshape(
            localization_map_max, shape=(B, 1, 1, 1, 1)
        )
            # Normalize the localization map.
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + 1e-6
        )
        localization_map = localization_map.data
        return localization_map



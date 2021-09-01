from .resnet import get_resnet_func_by_name
from .seresnet import get_seresnet_func_by_name
from .repvgg import get_RepVGG_func_by_name, repvgg_model_convert
from .conformer import get_conformer_func_by_name
from .mobilenetv3 import get_mobilenetv3_func_by_name

func_dict = {
    'resnet': get_resnet_func_by_name,
    'seresnet': get_seresnet_func_by_name,
    'mobilenetv3': get_mobilenetv3_func_by_name,
    'RepVGG': get_RepVGG_func_by_name,
    'Conformer': get_conformer_func_by_name
}


def build_network_by_name(name, pretrained, num_classes, **kwargs):

    if name.split("-")[0] not in func_dict.keys():
        raise Exception("An unsupported network type %s" % name)

    model_build_func = func_dict[name.split("-")[0]](name)
    net = model_build_func(num_classes=num_classes, pretrained=pretrained, **kwargs)

    return net

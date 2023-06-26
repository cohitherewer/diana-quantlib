import utils
import argparse
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.utils.serialization.Serializer import ModulesSerializer


parser = argparse.ArgumentParser("Generate an initial config.yaml file from a given model architecture")
parser.add_argument("model", choices=utils.all_models.keys(), help="Model architecture")
parser.add_argument("configfile", help="Name of the config file to generate (.yaml file)")
args = parser.parse_args()

# define model
model = utils.all_models[args.model]()
fq_model = DianaModule(DianaModule.from_trainedfp_model(model))
# instantiate serializer  and save path
serializer = ModulesSerializer(fq_model.gmodule)

# serialize
filepath = "config/resnet.yaml"
serializer.dump(args.configfile)

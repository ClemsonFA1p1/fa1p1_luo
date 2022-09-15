import os
from collections import OrderedDict
from torch.autograd import Variable
from translation.options.test_options import TestOptions
from translation.data.data_loader import CreateDataLoader
from translation.models.models import create_model
import translation.util.util as util
from translation.util.visualizer import Visualizer
from translation.util import html
import torch


class ImageTranslation():

    def __init__(self):
        self.opt = TestOptions().parse(save=False)
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip

        #data_loader = CreateDataLoader(self.opt)
        #dataset = data_loader.load_data()
        self.visualizer = Visualizer(self.opt)
        # create website
        self.web_dir = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch))
        #self.web_dir = os.path.join(self.opt.results_dir, exp_name, episode, self.opt.name))
        self.webpage = html.HTML(self.web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (self.opt.name, self.opt.phase, self.opt.which_epoch))

        # test
        if not self.opt.engine and not self.opt.onnx:
            self.model = create_model(self.opt)
            if self.opt.data_type == 16:
                self.model.half()
            elif self.opt.data_type == 8:
                self.model.type(torch.uint8)
                    
            #if self.opt.verbose:
            #    print(self.model)
        else:
            from run_engine import run_trt_engine, run_onnx

    def generate_image(self, data):
        #for i, data in enumerate(dataset):
        #if i >= self.opt.how_many:
        #    break
        #self.opt = Testself.options().parse(save=False)
        if self.opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif self.opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        #if self.opt.export_onnx:
        #    print ("Exporting to ONNX: ", self.opt.export_onnx)
        #    assert self.opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        #    torch.onnx.export(self.model, [data['label'], data['inst']],
        #                    self.opt.export_onnx, verbose=True)
        #    exit(0)
        minibatch = 1 
        if self.opt.engine:
            generated = run_trt_engine(self.opt.engine, minibatch, [data['label'], data['inst']])
        elif self.opt.onnx:
            generated = run_onnx(self.opt.onnx, self.opt.data_type, minibatch, [data['label'], data['inst']])
        else:        
            generated = self.model.inference(data['label'], data['inst'], data['image'])
        
        print(generated.shape)
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], self.opt.label_nc)),
                            ('synthesized_image', util.tensor2im(generated.data[0]))])
        #return generated
        img_path = data['path']
        print('process image... %s' % img_path)
        
        self.visualizer.save_images(self.webpage, visuals, img_path)
        return generated
        webpage.save()
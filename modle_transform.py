##### 查看 .pth.tar 文件模型参数 #####
# import torch
# import torchvision.models as models
#
# checkpoint = torch.load('weights/bytetrack_s_mot17.pth.tar')	# 加载模型
# print(checkpoint.keys())												# 查看模型元素
# state_dict = checkpoint['state_dict']
#
# print(checkpoint['epoch'])
# print(checkpoint['arch'])
# print(checkpoint['best_prec1'])
# # print(checkpoint['ce_optimizer'])
# print(state_dict.keys())
# print(state_dict['module.bn1.bias'])									# 打印 module.bn1.bias 的权值
# print(state_dict['module.bn1.bias'])									# 打印 module.bn1.bias 的形状
### 由 .pth.tar 文件加载模型 ####

# import torch
# import torchvision.models as models
#
# checkpoint = torch.load('weights/bytetrack_s_mot17.pth.tar')
# arch = checkpoint['model']
# model = models.__dict__[arch]()
# model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict(checkpoint['state_dict'])
# # print(model.state_dict())
# # print(model)
#
# #### save the model(only include parameters)
# torch.save(model.state_dict(), 'bytetrack_s_mot17.pth')
#
# torch.save(model, 'bytetrack_s_mot17.pth')

import torch
import torchvision
def get_model(self, sublinear=False):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    if "model" not in self.__dict__:
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        in_channels = [256, 512, 1024]
        # NANO model use depthwise = True, which is main difference.
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
        self.model = YOLOX(backbone, head)

    self.model.apply(init_yolo)
    self.model.head.initialize_biases(1e-2)
    return self.model


#### tar 转换为pth ####
state_dict = torch.load("weights/bytetrack_x_mot20.tar")
print(state_dict.keys()) # 查看模型元素
torch.save(state_dict,'bytetrack_s_mot20.pth')  # 保存整个模型
#### pth 转换为 pt ####

# 随机数字 batch_size, channels, width, height
model = get_model()   # 自己定义的网络模型
model.load_state_dict(torch.load('weights/bytetrack_s_mot20.pth'))
model.eval()

example = torch.rand(1,3,320,480)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")



# torch.save(model.state_dict(),'bytetrack_s_mot17.pth') # 只保存模型权重参数

# print(state_dict)
# model.load_state_dict(state_dict, False)
# model.eval()
#
# x = torch.rand(1, 3, 128, 128)
# ts = torch.jit.trace(model, x)
# ts.save('fcn_vgg16.net')



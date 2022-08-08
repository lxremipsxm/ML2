import torch
import torchvision.models as models

model = models.vgg16() # we do not specify pretrained=True
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

torch.save(model, 'model.pth')

model = torch.load('model.pth')
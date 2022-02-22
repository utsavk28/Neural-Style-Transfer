import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import numpy as np
from utils import build_loss,prepare_img,save_img,gram_matrix,prepare_model

def neural_style_transfer(content_img_path, content_img_name ,style_img_path ,style_img_name):
    init_method = 'random'
    height = 400
    weight_config = {
        'content_weight':1e5,
        'style_weight':3e4,
        'tv_weight':1e0,
    }
    
    output_img_dir = '../../images/output'
    out_dir_name = f'{content_img_name}-{style_img_name}'
    dump_path = os.path.join(output_img_dir, out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = prepare_img(content_img_path, height, device)
    style_img = prepare_img(style_img_path, height, device)

    if init_method == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif init_method == 'content':
        init_img = content_img
    else:
        style_img_resized = prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized
        
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(None, device)

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    num_of_iterations = 1000

    if True :
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations,
                                                                       content_feature_maps_index_name[0],
                                                                       style_feature_maps_indices_names[0], weight_config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'''L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={weight_config['content_weight'] * content_loss.item():12.4f},
                      style loss={weight_config['style_weight'] * style_loss.item():12.4f}, tv loss={weight_config['tv_weight'] * tv_loss.item():12.4f}''')
                save_img(optimizing_img, dump_path,out_dir_name,  cnt,num_of_iterations)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path

content_img_path = "../../images/input/content/"
content_img_name = ""
style_img_path = "../../images/input/style/"
style_img_name = ""

# neural_style_transfer(content_img_path,content_img_name,
#                       style_img_path ,style_img_name)

for i in os.listdir('../../images/input/content') :
    for j in os.listdir('../../images/input/style') :
        neural_style_transfer(f'{content_img_path}/{i}',i.split('.')[0],
                      f'{style_img_path}/{j}' ,j.split('.')[0])
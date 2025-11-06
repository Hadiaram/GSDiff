import sys

import cv2
from PIL import Image, ImageDraw

sys.path.append('/home/user00/HSZ/gsdiff-main')
sys.path.append('/home/user00/HSZ/gsdiff-main/datasets')
sys.path.append('/home/user00/HSZ/gsdiff-main/gsdiff')
sys.path.append('/home/user00/HSZ/gsdiff-main/scripts/metrics')



import math
import torch
import shutil
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from itertools import cycle
# from datasets.rplang_edge_semantics_simplified_55_106 import RPlanGEdgeSemanSimplified_55_106
# from datasets.rplang_edge_semantics_simplified import RPlanGEdgeSemanSimplified
from datasets.lifull_55_100 import *
from datasets.lifull import lifull
from gsdiff.heterhouse_75_106_lifull import HeterHouseModel
from gsdiff.heterhouse_56_13_lifull import EdgeModel
from gsdiff.utils import *
import torch.nn.functional as F
from scripts.metrics.fid import fid
from scripts.metrics.kid2 import kid2


diffusion_steps = 1000
batch_size_test = 1
device = 'cuda:0'
merge_points = True
align_points = True
aa_scale = 1
resolution = 512


'''create output_dir'''
output_dir = 'test_outputs/A-48-lifull-1/'
os.makedirs(output_dir, exist_ok=True)

'''Diffusion Settings'''
# cosine beta
alpha_bar = lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2
betas = []
max_beta = 0.999
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
betas = np.array(betas, dtype=np.float64)
alphas = 1.0 - betas  # [a1, a2, ..., a1000]: 1- -> 0+

# gaussian diffuse settings
alphas_cumprod = np.cumprod(alphas)  # [a1, a1a2, a1a2a3, ..., a1a2...a1000] 1- -> 0+
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # [1, a1, a1a2, a1a2a3, ..., a1a2...a999]
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)  # sqrt([a1, a1a2, a1a2a3, ..., a1a2...a1000]) 1- -> 0+
sqrt_one_minus_alphas_cumprod = np.sqrt(
    1.0 - alphas_cumprod)  # sqrt([1-a1, 1-a1a2, 1-a1a2a3, ..., 1-a1a2...a1000]) 0+ -> 1-
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)  # sqrt([1/a1, 1/a1a2, 1/a1a2a3, ..., 1/a1a2...a1000])
sqrt_recipm1_alphas_cumprod = np.sqrt(
    1.0 / alphas_cumprod - 1)  # sqrt([1/a1 - 1, 1/a1a2 - 1, 1/a1a2a3 - 1, ..., 1/a1a2...a1000 - 1])
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod)  # variance for posterior q(x_{t-1} | x_t, x_0)
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # posterior distribution mean
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (
            1.0 - alphas_cumprod)  # posterior distribution mean


'''Data'''
dataset_test = Lifull_55_100('test')
dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=8,
                        drop_last=False, pin_memory=True)  # try different num_workers to be faster
dataloader_test_iter = iter(cycle(dataloader_test))

dataset_test_for_gt_rendering = lifull('test')
dataloader_test_for_gt_rendering = DataLoader(dataset_test_for_gt_rendering, batch_size=batch_size_test, shuffle=False, num_workers=0,
                        drop_last=False, pin_memory=True)  # try different num_workers to be faster
dataloader_test_iter_for_gt_rendering = iter(cycle(dataloader_test_for_gt_rendering))


'''To compute FID and KID on the test set, the ground‑truth test graphs are first rasterized to images.'''
gt_dir_test = output_dir + 'test_gt' + '/'
if os.path.exists(gt_dir_test):
    shutil.rmtree(gt_dir_test)  # Remove existing directory
os.makedirs(gt_dir_test)  # Create directory
# Render each of the 3000 test samples to an image
# colors = {6: (0, 0, 0), 0: (222, 241, 244), 1: (159, 182, 234), 2: (92, 112, 107), 3: (95, 122, 224), 4: (123, 121, 95), 5: (143, 204, 242)}
# colors = {6: (0, 0, 0), 0: (244, 241, 222), 1: (234, 182, 159), 2: (107, 112, 92), 3: (224, 122, 95), 4: (95, 121, 123), 5: (242, 204, 143)}
colors = {1: (225, 255, 193),
          2: (254, 204, 203),
          3: (255, 246, 143),
          4: (191, 255, 255),
          5: (240, 255, 240),
          6: (249, 181, 106),
          7: (237, 121, 158),
          8: (232, 213, 154),
          9: (160, 232, 254),
          10: (46, 139, 87),
          11: (255, 255, 255),
          12: (0, 0, 0),
          0: (190, 190, 190)}
if len(dataset_test_for_gt_rendering) % batch_size_test == 0:
    batch_numbers = len(dataset_test_for_gt_rendering) // batch_size_test
else:
    batch_numbers = len(dataset_test_for_gt_rendering) // batch_size_test + 1
for batch_count in tqdm(range(batch_numbers)):
    corners_withsemantics_0_test_batch, global_attn_matrix_test_batch, corners_padding_mask_test_batch, edges_test_batch = next(dataloader_test_iter_for_gt_rendering)
    for i in range(corners_withsemantics_0_test_batch.shape[0]):
        test_count = batch_count * batch_size_test + i
        corners_withsemantics_0_test = corners_withsemantics_0_test_batch[i][None, :, :]
        global_attn_matrix_test = global_attn_matrix_test_batch[i][None, :, :]
        corners_padding_mask_test = corners_padding_mask_test_batch[i][None, :, :]
        edges_test = edges_test_batch[i][None, :, :]

        corners_withsemantics_0_test = corners_withsemantics_0_test.clamp(-1, 1).cpu().numpy()
        corners_0_test = (corners_withsemantics_0_test[0, :, :2] * (resolution // 2) + (resolution // 2)).astype(int)
        semantics_0_test = corners_withsemantics_0_test[0, :, 2:].astype(int)
        global_attn_matrix_test = global_attn_matrix_test.cpu().numpy()
        corners_padding_mask_test = corners_padding_mask_test.cpu().numpy()
        edges_test = edges_test.cpu().numpy()
        corners_0_test_depadded = corners_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 2)
        semantics_0_test_depadded = semantics_0_test[corners_padding_mask_test.squeeze() == 1][None, :, :]  # (n, 7)
        edges_test_depadded = edges_test[global_attn_matrix_test.reshape(1, -1, 1)][None, :, None]
        edges_test_depadded = np.concatenate((1 - edges_test_depadded, edges_test_depadded), axis=2)

        ''' get planar cycles'''
    # Array shape (1, n, 14) with 0/1; replace each 1 by its channel index, 0 by 99999 sentinel
        semantics_gt_i_transform_test = semantics_0_test_depadded
        semantics_gt_i_transform_indices_test = np.indices(semantics_gt_i_transform_test.shape)[-1]
        semantics_gt_i_transform_test = np.where(semantics_gt_i_transform_test == 1,
                                                 semantics_gt_i_transform_indices_test, 99999)

        gt_i_points_test = [tuple(corner_with_seman_test) for corner_with_seman_test in
                            np.concatenate((corners_0_test_depadded, semantics_gt_i_transform_test), axis=-1).tolist()[
                                0]]

        gt_i_edges_test = edges_to_coordinates(
            np.triu(edges_test_depadded[0, :, 1].reshape(len(gt_i_points_test), len(gt_i_points_test))).reshape(-1),
            gt_i_points_test)

        d_rev_test, simple_cycles_test, simple_cycles_semantics_test = get_cycle_basis_and_semantic_3_semansimplified_lifull(
            gt_i_points_test,
            gt_i_edges_test)


        # draw
        simple_cycles_test_aascale = []
        for polygon_i, polygon in enumerate(simple_cycles_test):
            polygon = [(p[0] * aa_scale, p[1] * aa_scale) for p in polygon]
            simple_cycles_test_aascale.append(polygon)
        simple_cycles_test = simple_cycles_test_aascale

        img = Image.new('RGB', (resolution * aa_scale, resolution * aa_scale), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for polygon_i, polygon in enumerate(simple_cycles_test):
            draw.polygon(polygon, fill=colors[simple_cycles_semantics_test[polygon_i]], outline=None)
        for polygon_i, polygon in enumerate(simple_cycles_test):
            for point_i, point in enumerate(polygon):
                if point_i < len(polygon) - 1:
                    p1 = (point[0], point[1])
                    draw.rectangle((p1[0] - round((3 if aa_scale == 1 else 3.5) * aa_scale), p1[1] - round((3 if aa_scale == 1 else 3.5) * aa_scale),
                                    p1[0] + round((3 if aa_scale == 1 else 3.5) * aa_scale), p1[1] + round((3 if aa_scale == 1 else 3.5) * aa_scale)), fill=(150, 150, 150), outline=None)
                    p2 = (polygon[point_i + 1][0], polygon[point_i + 1][1])
                    draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(150, 150, 150), width=7 * aa_scale)
        # img = img.resize((512, 512), Image.ANTIALIAS)
        img.save(os.path.join(gt_dir_test, f"test_gt_{test_count}.png"))


'''Neural Network'''
# Standalone training was used (faster and easier to control than joint training) and allows noise augmentation
model_path_EdgeModel = 'outputs/structure-56-34-interval1000/model_stage2_best_099000.pt'
model_EdgeModel = EdgeModel().to(device)
model_EdgeModel.load_state_dict(torch.load(model_path_EdgeModel, map_location=device))
# Freeze EdgeModel parameters
for param in model_EdgeModel.parameters():
    param.requires_grad = False

# Diffusion model
test_metrics = []
model_path_CDDPMs = ['outputs/structure-77-106-lifull/' + fn for fn in os.listdir('outputs/structure-77-106-lifull') if 'model' in fn and '.pt' in fn]
for model_path_CDDPM in model_path_CDDPMs:
    if int(model_path_CDDPM[5 + len('outputs/structure-77-106-lifull/'):8 + len('outputs/structure-77-106-lifull/')]) % 1000 == 100: # model1000000.pt
        model_CDDPM = HeterHouseModel().to(device)
        model_CDDPM.load_state_dict(torch.load(model_path_CDDPM, map_location=device))
    # Freeze EdgeModel parameters
        for param in model_CDDPM.parameters():
            param.requires_grad = False


        results_timesteps_stage1_test = [0]
        results_stage1_test = {}  # 999/900/.../0/'gt': (results, results_adjacency_lists)
        for k_test in results_timesteps_stage1_test:
            results_stage1_test['results_corners_' + str(k_test)] = []
            results_stage1_test['results_semantics_' + str(k_test)] = []
            results_stage1_test['results_corners_numbers_' + str(k_test)] = []
        if len(dataset_test) % batch_size_test == 0:
            batch_numbers = len(dataset_test) // batch_size_test
        else:
            batch_numbers = len(dataset_test) // batch_size_test + 1
        for batch_count in tqdm(range(batch_numbers)):
            '''a batch of data'''
            corners_withsemantics_0_test_batch, global_attn_matrix_test_batch, corners_padding_mask_test_batch, edges_test_batch = next(
                dataloader_test_iter)
            corners_withsemantics_0_test_batch = corners_withsemantics_0_test_batch.to(device).clamp(-1, 1)
            global_attn_matrix_test_batch = global_attn_matrix_test_batch.to(device)
            corners_padding_mask_test_batch = corners_padding_mask_test_batch.to(device)

            # (bs, 53, 10)
            corners_withsemantics_0_test_batch = torch.cat((corners_withsemantics_0_test_batch, (1 - corners_padding_mask_test_batch).type(corners_withsemantics_0_test_batch.dtype)), dim=2)

            '''reverse process: 999->998->...->1->0->final_pred(start)'''
            for current_step_test in list(range(diffusion_steps - 1, -1, -1)):
                if current_step_test == diffusion_steps - 1:
                    corners_withsemantics_t_test_batch = torch.randn(*corners_withsemantics_0_test_batch.shape,
                                                                     device=device,
                                                                     dtype=corners_withsemantics_0_test_batch.dtype)  # t=999
                else:
                    corners_withsemantics_t_test_batch = sample_from_posterior_normal_distribution_test_batch

                t_test = torch.tensor([current_step_test] * 1, device=device)

                '''model: predicts corners_noise'''
                output_corners_withsemantics1_test_batch, output_corners_withsemantics2_test_batch = model_CDDPM(corners_withsemantics_t_test_batch,
                                                                      global_attn_matrix_test_batch,
                                                                      t_test)
                output_corners_withsemantics_test_batch = torch.cat((output_corners_withsemantics1_test_batch, output_corners_withsemantics2_test_batch), dim=2)

                '''gaussian posterior'''
                model_variance_test_batch = torch.tensor(posterior_variance, device=device)[t_test][:, None,
                                            None].expand_as(
                    corners_withsemantics_t_test_batch)

                pred_xstart_test_batch = (
                        torch.tensor(sqrt_recip_alphas_cumprod, device=device)[t_test][:, None, None].expand_as(
                            corners_withsemantics_t_test_batch) * corners_withsemantics_t_test_batch -
                        torch.tensor(sqrt_recipm1_alphas_cumprod, device=device)[t_test][:, None, None].expand_as(
                            corners_withsemantics_t_test_batch) * output_corners_withsemantics_test_batch
                )  # to keep x0 always in (-1, 1) is unsuitable for which semantics number can be 2,3,...
                pred_xstart_test_batch[:, :, 0:2] = torch.clamp(pred_xstart_test_batch[:, :, 0:2], min=-1, max=1)
                pred_xstart_test_batch[:, :, 2:15] = pred_xstart_test_batch[:, :, 2:15] >= 0.5
                pred_xstart_test_batch[:, :, 15:16] = pred_xstart_test_batch[:, :, 15:16] >= 0.75

                model_mean_test_batch = (
                        torch.tensor(posterior_mean_coef1, device=device)[t_test][:, None, None].expand_as(
                            corners_withsemantics_t_test_batch) * pred_xstart_test_batch
                        + torch.tensor(posterior_mean_coef2, device=device)[t_test][:, None, None].expand_as(
                    corners_withsemantics_t_test_batch) * corners_withsemantics_t_test_batch
                )
                noise_test_batch = torch.randn_like(corners_withsemantics_t_test_batch)
                sample_from_posterior_normal_distribution_test_batch = model_mean_test_batch + torch.sqrt(
                    model_variance_test_batch) * noise_test_batch
                '''record to visualize'''
                if current_step_test in results_timesteps_stage1_test:
                    for i in range(corners_withsemantics_0_test_batch.shape[0]):
                        results_stage1_test['results_corners_' + str(current_step_test)].append(
                            sample_from_posterior_normal_distribution_test_batch[i, :, :2][None, :, :])
                        results_stage1_test['results_semantics_' + str(current_step_test)].append(
                            sample_from_posterior_normal_distribution_test_batch[i, :, 2:15][None, :, :])
                        results_stage1_test['results_corners_numbers_' + str(current_step_test)].append(
                                sample_from_posterior_normal_distribution_test_batch[i, :, 15:16][None, :, :].view(-1))  # 0: True, 1: False

        '''inverse normalize, remove padding and rasterization'''
        stage1_0_test = None
        for k_test in results_timesteps_stage1_test:
            result_corners_inverse_normalized_test, result_semantics_inverse_normalized_test = \
                inverse_normalize_and_remove_padding_100_4testing(results_stage1_test['results_corners_' + str(k_test)],
                                                     results_stage1_test['results_semantics_' + str(k_test)],
                                                     results_stage1_test['results_corners_numbers_' + str(k_test)],
                                                     resolution=resolution)
            stage1_0_test = (result_corners_inverse_normalized_test, result_semantics_inverse_normalized_test)

            # Compute average generated node count to assess whether complexity is too low
            node_count = 0


            # Visualization
            os.mkdir(output_dir + 'test_corner_' + 'step' + str(k_test) + '_' + model_path_CDDPM.split('/')[2].replace('.pt', ''))
            for i in tqdm(range(len(result_corners_inverse_normalized_test))):
                img = np.ones((resolution, resolution, 3), dtype=np.uint8)
                img *= 255
                # print(result_corners_inverse_normalized_test[i][0])
                for p in result_corners_inverse_normalized_test[i][0]:
                    cv2.circle(img, tuple(p.tolist()), 3, (random.randint(0, 220), random.randint(0, 220), random.randint(0, 220)), -1)
                cv2.imwrite(os.path.join(output_dir + 'test_corner_' + 'step' + str(k_test) + '_' + model_path_CDDPM.split('/')[2].replace('.pt', ''), f"{i}.png"), img)
                node_count += len(result_corners_inverse_normalized_test[i][0])
            node_count /= len(result_corners_inverse_normalized_test)
            print(node_count) #

        '''stage 2'''
        # merge points (data loading in actual)
        corners_all_samples_test = stage1_0_test[0]
        semantics_all_samples_test = stage1_0_test[1]
        if merge_points:
            corners_all_samples_merged_test = []
            semantics_all_samples_merged_test = []
            for i_test in range(len(dataset_test)):
                corners_i_test = corners_all_samples_test[i_test]
                semantics_i_test = semantics_all_samples_test[i_test]
                corners_merge_components_test = get_near_corners(corners_i_test, merge_threshold=resolution*0.01)
                indices_list_test = corners_merge_components_test
                corners_i_test = corners_i_test.reshape(-1, 2)
                semantics_i_test = semantics_i_test.reshape(-1, 13)
                full_indices_list_test = []
                for index_set_test in indices_list_test:
                    full_indices_list_test.extend(list(index_set_test))
                random_indices_list_test = []
                for index_set_test in indices_list_test:
                    random_index_test = random.choice(list(index_set_test))
                    random_indices_list_test.append(random_index_test)

                merged_corners_i_test = merge_array_elements(corners_i_test, full_indices_list_test,
                                                             random_indices_list_test)
                merged_semantics_i_test = merge_array_elements(semantics_i_test, full_indices_list_test,
                                                               random_indices_list_test)

                corners_all_samples_merged_test.append(merged_corners_i_test[None, :, :])
                semantics_all_samples_merged_test.append(merged_semantics_i_test[None, :, :])

            corners_all_samples_test = corners_all_samples_merged_test
            semantics_all_samples_test = semantics_all_samples_merged_test


        results_timesteps_stage2_test = [0]
        results_stage2_test = {}
        for k_test in results_timesteps_stage2_test:
            results_stage2_test['results_corners_' + str(k_test)] = []
            results_stage2_test['results_edges_' + str(k_test)] = []
            results_stage2_test['results_corners_numbers_' + str(k_test)] = []

        for test_count in tqdm(range(len(dataset_test))):
            '''a batch of data'''
            corners_stage2_test = torch.zeros((1, 53, 2), dtype=torch.float64, device=device)
            corners_temp_stage2_test = (torch.tensor(corners_all_samples_test[test_count], dtype=torch.float64,
                                                     device=device) - (resolution // 2)) / (resolution // 2)
            corners_stage2_test[:, 0:corners_temp_stage2_test.shape[1], :] = corners_temp_stage2_test

            semantics_stage2_test = torch.zeros((1, 53, 13), dtype=torch.float64, device=device)
            semantics_temp_stage2_test = torch.tensor(semantics_all_samples_test[test_count], dtype=torch.float64,
                                                      device=device)
            semantics_stage2_test[:, 0:semantics_temp_stage2_test.shape[1], :] = semantics_temp_stage2_test

            global_attn_matrix_stage2_test = torch.zeros((1, 53, 53), dtype=torch.bool, device=device)
            global_attn_matrix_stage2_test[:, 0:corners_temp_stage2_test.shape[1],
            0:corners_temp_stage2_test.shape[1]] = True
            corners_padding_mask_stage2_test = torch.zeros((1, 53, 1), dtype=torch.uint8, device=device)
            corners_padding_mask_stage2_test[:, 0:corners_temp_stage2_test.shape[1], :] = 1

            '''model: Edge transformer'''
            output_edges_test, _, _ = model_EdgeModel(corners_stage2_test, global_attn_matrix_stage2_test,
                                                corners_padding_mask_stage2_test, semantics_stage2_test)
            output_edges_test = F.softmax(output_edges_test, dim=2)
            output_edges_test = torch.argmax(output_edges_test, dim=2)
            output_edges_test = F.one_hot(output_edges_test, num_classes=2)

            '''record to visualize'''
            results_stage2_test['results_corners_' + str(0)].append(corners_stage2_test)
            results_stage2_test['results_edges_' + str(0)].append(output_edges_test)
            results_stage2_test['results_corners_numbers_' + str(0)].append(
                torch.sum(corners_padding_mask_stage2_test.view(-1)).item())

        '''inverse normalize, remove padding and visualization'''
        for k_test in [0]:
            edges_all_samples_test = edges_remove_padding(results_stage2_test['results_edges_' + str(k_test)],
                                                          results_stage2_test['results_corners_numbers_' + str(k_test)])

            output_dir_test = output_dir + 'test_' + model_path_CDDPM.split('/')[2].replace('.pt', '') + '/'
            if os.path.exists(output_dir_test):
                shutil.rmtree(output_dir_test)  # Remove existing directory
            os.makedirs(output_dir_test)  # Create directory
            for test_count in tqdm(range(len(dataset_test))):
                corners_sample_i_test = corners_all_samples_test[test_count]
                edges_sample_i_test = edges_all_samples_test[test_count]
                semantics_sample_i_test = semantics_all_samples_test[test_count]

                ''' get planar cycles'''
                # Array shape (1, n, 14) with 0/1; replace each 1 by its channel index, 0 by 99999 sentinel
                semantics_sample_i_transform_test = semantics_sample_i_test
                semantics_sample_i_transform_indices_test = np.indices(semantics_sample_i_transform_test.shape)[-1]
                semantics_sample_i_transform_test = np.where(semantics_sample_i_transform_test == 1,
                                                             semantics_sample_i_transform_indices_test, 99999)

                output_points_test = [tuple(corner_with_seman_test) for corner_with_seman_test in
                                      np.concatenate((corners_sample_i_test, semantics_sample_i_transform_test),
                                                     axis=-1).tolist()[0]]
                output_edges_test = edges_to_coordinates(
                    np.triu(edges_sample_i_test[0, :, 1].reshape(len(output_points_test), len(output_points_test))).reshape(
                        -1),
                    output_points_test)

                d_rev_test, simple_cycles_test, simple_cycles_semantics_test = get_cycle_basis_and_semantic_3_semansimplified_lifull(
                    output_points_test,
                    output_edges_test)
                # print(simple_cycles_test)
                # print(simple_cycles_semantics_test)
                '''save vector results for statistical analysis'''
                vr = {}
                vr['output_points_test'] = output_points_test
                vr['output_edges_test'] = output_edges_test
                vr['d_rev_test'] = d_rev_test
                vr['simple_cycles_test'] = simple_cycles_test
                vr['simple_cycles_semantics_test'] = simple_cycles_semantics_test
                np.save(output_dir + 'vr4stat_' + str(test_count) + '.npy', vr)






                
                if align_points:
                    align_threshold = round(resolution * 0.01)
                    # Strip trailing semantic labels after coordinates
                    # Build cleaned polygon list containing only valid (x,y) coordinate pairs
                    cleaned_polygons = []
                    # Iterate over each polygon
                    for polygon in simple_cycles_test:
                        cleaned_polygon = []
                        # Iterate vertices in polygon
                        for vertex in polygon:
                            # Keep only first two values (x,y)
                            cleaned_vertex = vertex[:2]
                            # Append to cleaned polygon
                            cleaned_polygon.append(cleaned_vertex)
                        # Add cleaned polygon to new dataset
                        cleaned_polygons.append(cleaned_polygon)
                    # print('2', cleaned_polygons)
                    # X‑axis band alignment
                    for x_bond_left in range(0, resolution - align_threshold):
                        x_bond_right = x_bond_left + align_threshold
                        # Collect edges whose endpoints fall inside this horizontal band
                        edges_inbond = []
                        for cp in cleaned_polygons:
                            for p_i, p in enumerate(cp):
                                if p_i < len(cp) - 1:
                                    e = (p, cp[p_i + 1])
                                    if x_bond_left <= e[0][0] <= x_bond_right and x_bond_left <= e[1][0] <= x_bond_right:
                                        edges_inbond.append(e)
                        # print('3', edges_inbond)
                    
                    
                        # Group edges by shared endpoints using a temporary networkx graph
                        G_inbond = nx.Graph()
                        # Add each geometric edge as a node
                        for edge_inbond in edges_inbond:
                            G_inbond.add_node(edge_inbond)
                        # Connect nodes whose underlying edges share a vertex
                        for edge1 in edges_inbond:
                            for edge2 in edges_inbond:
                                if edge1 != edge2:
                                    # Shared endpoint -> add graph edge
                                    if set(edge1) & set(edge2):
                                        G_inbond.add_edge(edge1, edge2)
                        # Extract connected components
                        connected_components_inbond = list(nx.connected_components(G_inbond))
                        # print('4', connected_components_inbond)
                    
                        # Convert components to edge lists
                        grouped_edge_sets = [list(component) for component in connected_components_inbond]
                    
                    
                        for i, component_edges in enumerate(grouped_edge_sets):  # For each edge group align X
                            # print(component_edges)
                            component_edges_vertices = []
                            for component_edges_t in component_edges:
                                for pnt in component_edges_t:
                                    component_edges_vertices.append(pnt)
                    
                            # print(f"Connected Component {i}: {component_edges}")
                            # Compute integer mean x of group and update member vertices
                            x_bar = round(sum([(t[0][0] + t[1][0]) for t in component_edges]) / (2 * len(component_edges)))
                    
                            # Rebuild cleaned polygon list with updated X values
                            cleaned_polygons_new = []
                            # Iterate over each polygon
                            for polygon in cleaned_polygons:
                                cleaned_polygon = []
                                # Iterate vertices
                                for vertex in polygon:
                                    if vertex in component_edges_vertices:
                                        # Replace X with aligned mean
                                        cleaned_vertex = (x_bar, vertex[1])
                                        # Append updated vertex
                                        cleaned_polygon.append(cleaned_vertex)
                                    else:
                                        # Keep original vertex
                                        cleaned_vertex = vertex
                                        # Append unchanged vertex
                                        cleaned_polygon.append(cleaned_vertex)
                                # Add updated polygon to new list
                                cleaned_polygons_new.append(cleaned_polygon)
                                # print(cleaned_polygon)
                            cleaned_polygons = cleaned_polygons_new
                    
                    # Y‑axis band alignment
                    for y_bond_up in range(0, resolution - align_threshold):
                        y_bond_down = y_bond_up + align_threshold
                        # Collect edges whose endpoints fall inside this vertical band
                        edges_inbond = []
                        for cp in cleaned_polygons:
                            for p_i, p in enumerate(cp):
                                if p_i < len(cp) - 1:
                                    e = (p, cp[p_i + 1])
                                    if y_bond_up <= e[0][1] <= y_bond_down and y_bond_up <= e[1][1] <= y_bond_down:
                                        edges_inbond.append(e)
                        # print(edges_inbond)
                        # Group edges by shared endpoints using a temporary networkx graph
                        G_inbond = nx.Graph()
                        # Add each geometric edge as a node
                        for edge_inbond in edges_inbond:
                            G_inbond.add_node(edge_inbond)
                        # Connect nodes if underlying edges share a vertex
                        for edge1 in edges_inbond:
                            for edge2 in edges_inbond:
                                if edge1 != edge2:
                                    # Shared endpoint -> add graph edge
                                    if set(edge1) & set(edge2):
                                        G_inbond.add_edge(edge1, edge2)
                        # Extract connected components
                        connected_components_inbond = list(nx.connected_components(G_inbond))
                        # Convert components to edge lists
                        grouped_edge_sets = [list(component) for component in connected_components_inbond]
                        for i, component_edges in enumerate(grouped_edge_sets):  # For each edge group align Y
                            # print(component_edges)
                            component_edges_vertices = []
                            for component_edges_t in component_edges:
                                for pnt in component_edges_t:
                                    component_edges_vertices.append(pnt)
                            # print(f"Connected Component {i}: {component_edges}")
                            # Compute integer mean y of group and update member vertices
                            y_bar = round(sum([(t[0][1] + t[1][1]) for t in component_edges]) / (2 * len(component_edges)))
                            # Rebuild cleaned polygon list with updated Y values
                            cleaned_polygons_new = []
                            # Iterate over each polygon
                            for polygon in cleaned_polygons:
                                cleaned_polygon = []
                                # Iterate vertices
                                for vertex in polygon:
                                    if vertex in component_edges_vertices:
                                        # Replace Y with aligned mean
                                        cleaned_vertex = (vertex[0], y_bar)
                                        # Append updated vertex
                                        cleaned_polygon.append(cleaned_vertex)
                                    else:
                                        # Keep original vertex
                                        cleaned_vertex = vertex
                                        # Append unchanged vertex
                                        cleaned_polygon.append(cleaned_vertex)
                                # Add updated polygon to new list
                                cleaned_polygons_new.append(cleaned_polygon)
                            cleaned_polygons = cleaned_polygons_new
                    simple_cycles_test = cleaned_polygons
                    




                
                # draw
                simple_cycles_test_aascale = []
                for polygon_i, polygon in enumerate(simple_cycles_test):
                    polygon = [(p[0] * aa_scale, p[1] * aa_scale) for p in polygon]
                    simple_cycles_test_aascale.append(polygon)
                simple_cycles_test = simple_cycles_test_aascale

                img = Image.new('RGB', (resolution * aa_scale, resolution * aa_scale), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                for polygon_i, polygon in enumerate(simple_cycles_test):
                    draw.polygon(polygon, fill=colors[simple_cycles_semantics_test[polygon_i]], outline=None)
                for polygon_i, polygon in enumerate(simple_cycles_test):
                    for point_i, point in enumerate(polygon):
                        if point_i < len(polygon) - 1:
                            p1 = (point[0], point[1])
                            draw.rectangle((p1[0] - round((3 if aa_scale == 1 else 3.5) * aa_scale), p1[1] - round((3 if aa_scale == 1 else 3.5) * aa_scale),
                                            p1[0] + round((3 if aa_scale == 1 else 3.5) * aa_scale), p1[1] + round((3 if aa_scale == 1 else 3.5) * aa_scale)),
                                           fill=(150, 150, 150), outline=None)
                            p2 = (polygon[point_i + 1][0], polygon[point_i + 1][1])
                            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(150, 150, 150), width=7 * aa_scale)
                # img = img.resize((512, 512), Image.ANTIALIAS)
                img.save(os.path.join(output_dir_test, f"test_pred_{test_count}.png"))

        '''calculate FID, KID. '''
        current_Fid = fid(gt_dir_test, output_dir_test, fid_batch_size=128, fid_device=device)
        current_Kid = kid2(gt_dir_test, output_dir_test, kid_batch_size=128, kid_device=device)
        print(model_path_CDDPM, 'FID: ', current_Fid, 'KID: ', current_Kid)

        '''saving Acc'''
        test_metrics.append([model_path_CDDPM, current_Fid, current_Kid])
        np.save(output_dir + 'test_metrics.npy', np.array(test_metrics))

import os

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from common_utils.image import imread
from common_utils.common import two_tuple
from common_utils.resize_right import resize
from common_utils.video import torchvid2mp4
from common_utils.robot_pose import read_pose_from_file
# from common_utils.image import imread
from common_utils.video import read_frames_from_dir

from config import *
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion import Diffusion
from diffusion.diffusion_utils import save_diffusion_sample
from models.nextnet import NextNet
from torchvision import transforms




def get_model_path(image_name, version_name):
    return os.path.join('lightning_logs', image_name, version_name, 'checkpoints', 'last.ckpt')


def create_sample_directory(cfg, extra_path=''):
    sample_directory = os.path.join(cfg.output_dir, cfg.image_name, cfg.run_name, extra_path)
    os.makedirs(sample_directory, exist_ok=True)
    print(f'Sample directory: {sample_directory}')
    return sample_directory


def noise_img(img, model, t):
    """
    Add noise (equivalent to t steps of a forward diffusion process) to an image.

    Args:
        img (torch.Tensor): Image to add noise to.
        model (Diffusion or ConditionalDiffusion): Diffusion model with "q_sample" implementation.
        t (int): Number of forward diffusion steps to perform.
    """
    batch_size = img.shape[0]
    if isinstance(model, Diffusion):
        noisy_img = model.q_sample(img, t)
    elif isinstance(model, ConditionalDiffusion):
        continuous_sqrt_alpha_hat = torch.FloatTensor(np.random.uniform(model.sqrt_alphas_hat_prev[t - 1], model.sqrt_alphas_hat_prev[t], size=batch_size)).to(img.device).view(batch_size, -1)
        noisy_img = model.q_sample(img, continuous_sqrt_alpha_hat.view(-1, 1, 1, 1))
    else:
        raise Exception

    return noisy_img


"""
Note as the pose_diff relies on the number of robot poses used for training, so 
when sampling, 

start_frame_index + current_frame_index should be smaller than the cfg.output_video_len

in our case, 

start_frame_index = 33
len(robot_poses) = 91

so 

cfg.output_video_len <= 91-33 -1 = 57 

if 58 there will be a out of index issue

"""
def sample_pose_diff(condition_frame, start_frame_index = None, current_frame_index = None, robot_poses = None, frame_diff = 1):
    frame_to_generate_id = start_frame_index + current_frame_index
    conditioned_frame_id = frame_to_generate_id - frame_diff

    frame_to_generate_pose = robot_poses[frame_to_generate_id]
    conditioned_frame_pose = robot_poses[conditioned_frame_id]

    pose_diff = frame_to_generate_pose - conditioned_frame_pose

    pose_diff = torch.from_numpy(pose_diff).float().to(condition_frame.device).view(1,-1)

    return pose_diff, conditioned_frame_id, frame_to_generate_id


def generate_video(cfg):
    """
    Generates and saves a video (in mp4 format).

    Special parameters that cfg argument should include:
        frame_size (int or tuple(int, int)):
            The size of the frames. If None, the original spatial size of the video frames will be used.
        start_frame_index (int):
            Which frame of the video to start the generation from. If None, the first frame will be generated from noise
            by the DDPM frame Projector.
    """
    sample_directory = create_sample_directory(cfg, 'frames')
    # robot_poses = read_pose_from_file(f'./robot_data/robot_pose_old.txt')
    robot_poses = read_pose_from_file(f'./robot_data/ur5_data_rad_extrapolate.txt')
    masks = read_frames_from_dir(f'./images/video/{cfg.mask_name}') 
    # mask_transform = transforms.Compose([
            # transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
            # transforms.Lambda(lambda img: (img[:3, ] / 255.0 * 2.0) - 1.0)])
    print(masks.shape)
    print('===================================================================')
    # Load models
    predictor_path = get_model_path(cfg.image_name, cfg.run_name + '_predictor')
    predictor_model = ConditionalDiffusion.load_from_checkpoint(predictor_path, training_target='noise',
                                                      model=NextNet(in_channels=6, depth=cfg.network_depth, frame_conditioned=True, mask_channels=6),
                                                      noise_schedule='cosine', timesteps=cfg.diffusion_timesteps).cuda()

    projector_path = get_model_path(cfg.image_name, cfg.run_name + '_projector')
    
    #load_from_checkpoint is inherited from the LightningModule
    projector_model = Diffusion.load_from_checkpoint(projector_path, training_target='noise', model=NextNet(depth=16, mask_channels=3),
                                                 noise_schedule='cosine', timesteps=cfg.diffusion_timesteps).cuda()

    video_dir = os.path.join('.', 'images', 'video', f'{cfg.image_name}')

    # Choose starting frame
    if cfg.start_frame_index is None:
        frame_shape = imread(os.path.join(video_dir, f'1.png')).shape[-2:]
        start_frame = projector_model.sample(image_size=frame_shape, batch_size=1)
    else:
        start_frame = imread(os.path.join(video_dir, f'{cfg.start_frame_index}.png')).cuda() * 2 - 1

    if cfg.sample_size is not None:
        start_frame = resize(start_frame, out_shape=cfg.sample_size)

    save_diffusion_sample(start_frame, os.path.join(sample_directory, '0.png'))

    # Sample frames
    # start_frame = mask_transform(start_frame)
    samples = [start_frame]

    # correct_t is 3
    correction_t = {50: 3, 500: 10}.get(cfg.diffusion_timesteps, cfg.diffusion_timesteps / 50)
    # output_video_len is 100
    for frame in range(1, cfg.output_video_len + 1):
        print(frame)

        # print(f'++++++++++{cfg.start_frame_index + frame}')
        # Sample the next frame
        # frame_diff is 1
        # The next frame is generated by conditioning on the most recently generated frame
        # The default frame_diff is 1, so basically conditioned on the previous image to generate a new one
        # The start_frame_index needs to be considered when generate pose_diff
        pose_diff, condition_frame_id, frame_to_generated_id = sample_pose_diff(samples[-1], start_frame_index = cfg.start_frame_index, current_frame_index=frame, robot_poses = robot_poses, frame_diff=cfg.frame_diff)
        
        condition_mask=masks[condition_frame_id,:,:,:].to(predictor_model.device).unsqueeze(0)
        # condition_mask = mask_transform(condition_mask)
        to_generate_mask=masks[frame_to_generated_id,:,:,:].to(predictor_model.device).unsqueeze(0)
        # to_generate_mask = mask_transform(to_generate_mask)
        mask = torch.cat([condition_mask, to_generate_mask], dim=1)
        # print(f'--mask shape = {mask.shape} --')
        # print(f'--mask device = {mask.device} --')

        # print(f'====predictor_model = {predictor_model.device} ====')
        next_frame = predictor_model.sample(condition=samples[-1], pose_diff = pose_diff, frame_diff=cfg.frame_diff, mask=mask)

        # Correct the sampled frame
        noisy_next_frame = noise_img(next_frame, projector_model, correction_t)
        corrected_next_frame = projector_model.sample(custom_initial_img=noisy_next_frame, custom_timesteps=correction_t, mask=to_generate_mask) # needs look into
        
        samples.append(corrected_next_frame)

        save_diffusion_sample(corrected_next_frame, os.path.join(sample_directory, f'{frame}.png'))

    # Save video
    ordered_samples = torch.cat(samples, dim=0)
    resized_samples = resize(ordered_samples, out_shape=(3, (start_frame.shape[-2] // 2) * 2, (start_frame.shape[-1] // 2) * 2))
    torchvid2mp4(resized_samples.permute((1, 0, 2, 3)), os.path.join(sample_directory, '..', 'generated.mp4'), fps=20)


def interpolate_video(cfg):
    """
    Performs temporal upsampling on a video and saves the result.
    Special parameters that cfg argument should include:
        interpolation_rate (int):
            Factor by which the video length will be increased (e.g. 4 -> 4x temporal upsampling).
    """
    sample_directory = create_sample_directory(cfg, 'frames')

    projector_path = get_model_path(cfg.image_name, cfg.run_name + '_projector')
    projector_model = Diffusion.load_from_checkpoint(projector_path, training_target='noise',
                                                     model=NextNet(depth=cfg.network_depth),
                                                     noise_schedule='cosine', timesteps=cfg.diffusion_timesteps).cuda()

    interpolate_path = get_model_path(cfg.image_name, cfg.run_name + '_interpolator')
    interpolate_model = ConditionalDiffusion.load_from_checkpoint(interpolate_path, training_target='x0',
                                                                  model=NextNet(in_channels=9, depth=cfg.network_depth),
                                                                  noise_schedule='cosine',
                                                                  timesteps=cfg.diffusion_timesteps).cuda()

    video_dir = os.path.join('.', 'images', 'video', f'{cfg.image_name}')
    original_frame_count = len(os.listdir(video_dir))
    samples = [imread(os.path.join(video_dir, f'{frame_idx}.png')).cuda() * 2 - 1 for frame_idx in range(1, original_frame_count + 1)]

    # Set T_{corr} for the projector correction noise
    correction_t = {50: 3, 500: 10}.get(cfg.diffusion_timesteps, cfg.diffusion_timesteps / 50)
    cur_interpolation = 2
    while cur_interpolation <= cfg.interpolation_rate:
        print("Samples is currently in length: ", len(samples))
        print('Current interpolation: ', cur_interpolation)
        new_samples = [samples[0]]
        for i in range(len(samples) - 1):
            new_frame = interpolate_model.sample(condition=torch.cat([samples[i], samples[i + 1]], dim=1))

            # Correct the sampled frame
            noisy_next_frame = noise_img(new_frame, projector_model, correction_t)
            corrected_next_frame = projector_model.sample(custom_initial_img=noisy_next_frame, custom_timesteps=correction_t)
            new_samples.append(corrected_next_frame)
            new_samples.append(samples[i + 1])
        samples = new_samples.copy()
        cur_interpolation *= 2

    for i, s in enumerate(samples):
        save_diffusion_sample(s, os.path.join(sample_directory, f'{i + 1}.png'))

    # Save video
    ordered_samples = torch.cat(samples, dim=0)
    resized_samples = resize(ordered_samples, out_shape=(3, (samples[0].shape[-2] // 2) * 2, (samples[0].shape[-1] // 2) * 2))
    torchvid2mp4(resized_samples.permute((1, 0, 2, 3)), os.path.join(sample_directory, '..', f'generated.mp4'), fps=10 * cfg.interpolation_rate)


def generate_diverse_samples(cfg):
    """
    Generates diverse image samples from a single image DDPM trained model.

    Args:
        cfg (Config):
            Configuration object.
    """
    # Create sample directory
    sample_directory = create_sample_directory(cfg)

    # Load model
    path = get_model_path(cfg.image_name, cfg.run_name)
    model = Diffusion.load_from_checkpoint(path, model=NextNet(depth=cfg.network_depth),
                                           timesteps=cfg.diffusion_timesteps,
                                           training_target='x0',
                                           noise_schedule='linear').cuda()

    if cfg.sample_size is None:
        size = tuple(imread(f'./images/{cfg.image_name}').shape[-2:])
    else:
        size = two_tuple(cfg.sample_size)

    # Sample and save images
    batch_size = 8
    samples = []
    for i in tqdm(range(0, cfg.sample_count, batch_size)):
        samples.append(model.sample(image_size=size, batch_size=min(batch_size, cfg.sample_count - i)))
    samples = torch.cat(samples, dim=0)
    save_diffusion_sample(samples, os.path.join(sample_directory, 'sample.png'))


def main():
    cfg = Config()
    cfg = parse_cmdline_args_to_config(cfg)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    if cfg.task == 'video':
        generate_video(cfg)
    elif cfg.task == 'video_interp':
        interpolate_video(cfg)
    else:
        generate_diverse_samples(cfg)


if __name__ == '__main__':
    main()

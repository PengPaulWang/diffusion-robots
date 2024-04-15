
# Diffusion - Robot Shape and Location Retainment

Diffusion models have marked a significant milestone in the enhancement of image and video generation technologies. However, generating videos that precisely retain the shape and location of moving objects such as robots remains a challenge. This paper presents diffusion models specifically tailored to generate videos that accurately maintain the shape and location of mobile robots. This development offers substantial benefits to those working on detecting dangerous interactions between humans and robots by facilitating the creation of training data for collision detection models, circumventing the need for collecting data from the real world, which often involves legal and ethical issues. Our models incorporate techniques such as embedding accessible robot pose information and applying semantic mask regulation within the ConvNext backbone network. These techniques are designed to refine intermediate outputs, therefore improving the retention performance of shape and location. Through extensive experimentation, our models have demonstrated notable improvements in maintaining the shape and location of different robots, as well as enhancing overall video generation quality, compared to the benchmark diffusion model.


[Supplemental Materials](https://stummuac-my.sharepoint.com/personal/55141653_ad_mmu_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F55141653%5Fad%5Fmmu%5Fac%5Fuk%2FDocuments%2FFaculty%2Ddoc%2FResearch%2FPeng%2DWang%2FIROS%5F2024%2FSupplemental%20materials%20to%20%20Robot%20Shape%20and%20Location%20Retention%20in%20Video%20Generation%20Using%20Diffusion%20Models%2Epdf&parent=%2Fpersonal%2F55141653%5Fad%5Fmmu%5Fac%5Fuk%2FDocuments%2FFaculty%2Ddoc%2FResearch%2FPeng%2DWang%2FIROS%5F2024&ga=1)


## Data


As mentioned in the paper, super-resolution or generating high-resolution frames is beyond the scope of this paper. We aim at generating frames in which the shape and the location of objects of interest (robots in our case) are retained as accurate as possible. Therefore, regardless of the original sizes of the video frames, we have resized both the frames and masks to dimensions of 256x144 to save computational resources as well as accelerate the training and sampling process. Our dataset is publicly accessible on [Data on Roboflow](https://app.roboflow.com/turtlebot-h8awt).


## Requirements & Running


We have built our models on top of SinFusion, which also serves as the benchmark model for comparison. We take the opportunity here to show our gratitude to the authors of [SinFusion](https://github.com/yanivnik/sinfusion-code). For convenience, we have integrated our codes into SinFusion. We hope this integration does not mislead readers into overlooking our contributions.

For clarity, we have created codes for each model proposed and using different suffixes, e.g., 'conditional_diffusion_mask_only.py' indicates that only masks are used as conditions; 'conditional_diffusion_robot_pose.py' indicates that only robot pose is used as conditions; 'conditional_diffusion_sinFusion.py' indicates that it is the original SinFusion codes and no conditions are used;  and 'conditional_diffusion.py' indicates that both robot poses and masks are used as conditions. Please note to run a certain model, one needs to delete the suffix. This means 'conditional_diffusion.py' that uses both robot poses and masks as conditions are the current model running.

Similar rules apply to other codes as well.

A JupyterNote book is also provided to run the codes. All codes were ran with A100 GPU on Google Colab.


## Potential Problems
There are no known issues with the codes, but if you encouter any, please feel free to contact [Peng Wang](https://www.mmu.ac.uk/staff/profile/dr-peng-wang) at p.wang@mmu.ac.uk

Considering different running environments, it is normal for the reimplementation results to be different from those in the paper. Please contact  [Peng Wang](https://www.mmu.ac.uk/staff/profile/dr-peng-wang) at p.wang@mmu.ac.uk for any issues.

## BiblioTeX

```
  @misc{peng2024diffusion,
        title={Robot Shape and Location Retention in Video Generation Using Diffusion Models}, 
        author={Peng Wang, Zhihao Guo, Abdul Sait, and Minh Huy Pham},
        year={2024},
        eprint={xx},
        archivePrefix={arXiv},
        primaryClass={cs.RO}
  }
```

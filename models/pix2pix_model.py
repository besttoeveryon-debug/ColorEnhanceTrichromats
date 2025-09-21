import torch
from .base_model import BaseModel
from . import networks
import numpy
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from skimage import feature
import os
import torchvision.utils as vutils


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')

        return parser
  

    def __init__(self, opt):
        """Initialize the Pix2Pix-based model for red-green color vision deficiency enhancement.

        This function sets up the generator and discriminator networks, loss functions, and optimizers
        for training a model to enhance images for red-green color vision deficiency (CVD). It uses a
        Machado matrix to simulate protanopia or deuteranopia and supports perceptual, edge, and GAN losses.
    
        Parameters:
            opt (Option class): Configuration object storing experiment flags, must be a subclass of BaseOptions.
                               Contains parameters like input_nc, output_nc, netG, netD, lr, etc.
    
        Attributes:
            matrix (torch.Tensor): yang matrix for simulating red-green CVD (protanopia, shape: [3, 3]).
            loss_names (list): List of loss names to track during training (e.g., D_real, G_perceptual).
            visual_names (list): List of image names to save/display (e.g., real_A, fake_B).
            model_names (list): List of model names to save/load (e.g., G, D).
            netG (torch.nn.Module): Generator network for image enhancement.
            netD (torch.nn.Module): Discriminator network (used during training).
            criterionGAN (callable): GAN loss function (e.g., vanilla or LSGAN).
            criterionL1 (callable): L1 loss function for pixel-wise comparison.
            optimizer_G (torch.optim.Optimizer): Adam optimizer for the generator.
            optimizer_D (torch.optim.Optimizer): Adam optimizer for the discriminator.
            optimizers (list): List of optimizers for training.
    
        Notes:
            - The Machado matrix is based on Machado et al. (2009) for protanopia simulation (Psimulated11).
            - Other matrix options (e.g., Dsimulated6, Dsimulated11, Psimulated6) are commented out for flexibility.
            - The training objective includes GAN loss, perceptual loss, edge loss, and simulation loss.
            - CBAM (Convolutional Block Attention Module) is optionally disabled via opt.use_cbam.
        """
        BaseModel.__init__(self, opt)
        # Yang matrix for protanopia simulation (Psimulated11)
        self.matrix = torch.tensor([[0.3802, 0.4192, 0.2007],
                                    [0.0812, 0.9451, -0.0263],
                                    [0.0031, -0.0021, 0.9990]], dtype=torch.float32).to(self.device)
        # Alternative matrices (commented out for reference)
        # Dsimulated6: [[0.7952, 0.1957, 0.0091], [0.0862, 0.9176, -0.0038], [-0.0077, 0.0073, 1.0003]]
        # Dsimulated11: [[0.5760, 0.3548, 0.0692], [0.1785, 0.8506, -0.0292], [-0.0159, 0.0133, 1.0026]]
        # Psimulated6: [[0.6458, 0.2263, 0.1278], [0.0464, 0.9704, -0.0167], [0.0018, -0.0011, 0.9994]]
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real', 'D_fake', 'G_perceptual', 'G_sim' , 'edge' , 'G_GAN']# 'G_L1'
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B','simulated_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,not opt.use_cbam)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)    
    def colorblind_simulator(self,image):
        """Simulate red-green color vision deficiency using the Machado matrix.
    
        This function applies a transformation matrix (e.g., Psimulated11 from Machado et al., 2009)
        to simulate the visual perception of red-green color vision deficiency (protanopia or deuteranopia).
        The process includes sRGB linearization, matrix transformation, and inverse linearization.
    
        Parameters:
            image (torch.Tensor): Input RGB image tensor, shape [B, C, H, W], values in [-1, 1].
    
        Returns:
            torch.Tensor: Simulated image tensor with color vision deficiency, shape [B, C, H, W],
                          values in [-1, 1].
    
        Notes:
            - The transformation uses the Machado matrix (Psimulated11 for protanopia by default).
            - sRGB linearization follows the standard sRGB gamma correction (ITU-R BT.709).
            - The output is clamped to ensure pixel values remain in [-1, 1] for consistency with the model.
            - Reference: Machado et al., "Computing and Visualizing Color Deficiencies," 2009.
        """
        # Linearize sRGB (convert input from [-1, 1] to [0, 1])
        image=(image + 1) /2
        idx = image <= 0.04045
        idy = image > 0.04045
        image_linear = image.clone()
        image_linear[idx] = image[idx] / 12.92
        image_linear[idy] = ((image[idy] + 0.055) / 1.055) ** 2.4


         # Reshape and apply Machado matrix transformation
        image_linear = image_linear.permute(0, 2, 3, 1)  # [B, H, W, C]
        simulated = torch.matmul(image_linear, self.matrix.T)  # Simulate CVD with matrix multiplication
        simulated = simulated.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Inverse sRGB linearization (convert back to [0, 1])
        idx1 = simulated <= 0.0031308
        idy1 = simulated > 0.0031308
        simulated[idx1] = simulated[idx1] * 12.92
        simulated[idy1] = 1.055 * (simulated[idy1] ** (1 / 2.4)) - 0.055
        simulated = torch.clamp(simulated, 0, 1)
        # Convert back to [-1, 1] for model compatibility
        simulated = simulated * 2 - 1

        return simulated
    def edge_loss(self, fake, real):
        """Compute edge loss between generated and real images using Sobel filters.
    
        This function calculates the mean absolute difference between edges extracted from
        generated and real images, using Sobel filters to detect horizontal and vertical edges.
        The loss encourages the generated images to preserve edge structures, which is critical
        for red-green color vision deficiency (CVD) enhancement.
    
        Parameters:
            fake (torch.Tensor): Generated image tensor, shape [B, C, H, W], values in [-1, 1].
            real (torch.Tensor): Real image tensor, shape [B, C, H, W], values in [-1, 1].
    
        Returns:
            torch.Tensor: Scalar edge loss value, weighted by a factor of 10.
    
        Notes:
            - Sobel filters are applied separately to x and y directions, with group convolution to
              process each RGB channel independently.
            - The loss is computed as the mean absolute difference between edge magnitudes.
            - The weighting factor (10) balances the edge loss with other losses (e.g., TCC, CD, GAN).
            - This loss is particularly important for CVD enhancement to ensure structural fidelity.
        """
        # Define Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).cuda()
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Compute edges for generated image
        fake_edge_x = torch.nn.functional.conv2d(fake, sobel_x, groups=3, padding=1)
        fake_edge_y = torch.nn.functional.conv2d(fake, sobel_y, groups=3, padding=1)
        fake_edge = torch.sqrt(fake_edge_x**2 + fake_edge_y**2)

        # Compute edges for real image
        real_edge_x = torch.nn.functional.conv2d(real, sobel_x, groups=3, padding=1)
        real_edge_y = torch.nn.functional.conv2d(real, sobel_y, groups=3, padding=1)
        real_edge = torch.sqrt(real_edge_x**2 + real_edge_y**2)

        # Compute mean absolute edge difference, scaled by weight
        return torch.mean(torch.abs(fake_edge - real_edge)) * 10

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B = torch.clamp(self.fake_B, -1, 1)
        self.simulated_B = self.colorblind_simulator(self.fake_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        epoch_str = self.opt.epoch
        #绿11
        #self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 1
        #红6
        #self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 1.5
        #红11
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 1.5
        self.perceptual_loss = networks.PerceptualLoss()
        self.loss_G_perceptual = self.perceptual_loss(self.fake_B, self.real_B)
        self.loss_G_sim = self.criterionL1(self.simulated_B, self.real_A) 
        # Dynamic weighting based on epoch
        weight_sim = 50
        #红6
        #weight_perceptual = 2
        #绿11
        #weight_perceptual = 1
        #红11
        weight_perceptual = 2
        # 应用权重
        #self.loss_G_sim = self.loss_G_sim * weight_sim
        self.loss_G_sim = self.loss_G_sim * weight_sim
        self.loss_G_perceptual = self.loss_G_perceptual * weight_perceptual
        #绿11
        #self.loss_edge = self.edge_loss(self.fake_B, self.real_A) * 15
        #红6
        #self.loss_edge = self.edge_loss(self.fake_B, self.real_A) * 30
        #红11
        self.loss_edge = self.edge_loss(self.fake_B, self.real_A) * 25
    
        # Combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_perceptual + self.loss_G_sim + self.loss_edge
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

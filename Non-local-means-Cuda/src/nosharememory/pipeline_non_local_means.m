%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in Cuda GPu.
%
% Mastoras Rafail Evangelos
% 7918
% January 2017
%
% DEPENDENCIES
%
% [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%     algorithm for image denoising. In 2005 IEEE Computer Society
%     Conference on Computer Vision and Pattern Recognition (CVPR’05),
%      volume 2, pages 60–65. IEEE, 2005.
%

  clear all %#ok
  close all

  %% PARAMETERS
  fprintf('This is version does not use shared memory\nthus it will be slow for images more than 512x512\n');
  
  fprintf('If u want to check the TEST image enter resolution 1 (image is 64x64)\n');
  in = input('State image resolution(64,128,256,512,1024,2048,4096 or 1 for TEST image): ');
    while(1)
        if (in~=64) && (in~=128) && (in~=256) && (in~=512) && (in~=1024) && (in~=2048) && (in~=4096) && (in~=1)
            disp('The input must be 64,128,256,512,1024,2048,4096 or 1 for TEST image');
            in = input('State the Image width: ');
        else
            break;
        end
    end
 % input image   
  if in==64
      pathImg   = '../data/house.mat';
      strImgVar = 'house';
  elseif in==128
      pathImg   = '../data/cafe.mat';
      strImgVar = 'cafe';
  elseif in==256
      pathImg   = '../data/einstein.mat';
      strImgVar = 'einstein';
  elseif in==512 
      pathImg   = '../data/flinstones.mat';
      strImgVar = 'flinstones';
  elseif in==1024
      pathImg   = '../data/moon.mat';
      strImgVar = 'moon';
  elseif in==2048
      pathImg   = '../data/lion2.mat';
      strImgVar = 'lion2';
  elseif in==4096
      pathImg   = '../data/chameleon.mat';
      strImgVar = 'chameleon';
  else
      pathImg   = '../data/test.mat';
      strImgVar = 'test';
  end
  
  patchSize = input('State the patch size(3,5 or 7): ');
    while(1)
        if (patchSize~=3) && (patchSize~=5) && (patchSize~=7)
            disp('The patch size must be 3,5 or 7');
            patchSize = input('State the patch size(3,5 or 7): ');
        else
            break;
        end
    end
  
  
  
  %% noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  

  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);

  %% INPUT DATA

  fprintf('...loading input data...\n')

  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);

  %% PREPROCESS

  fprintf('...normalizing image...\n')
  I = normImg( I );

  %% NOISE

  fprintf('...applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  
  [m n]=size(J);
  %% filter sigma value
  if patchSize==3 && m==64
    filtSigma=0.013;
  else
    filtSigma=0.0225;
  end
  
  patchSigma=5/3;
  H = fspecial('gaussian',[patchSize patchSize], patchSigma);
  H = H(:) ./ max(H(:));
  H=H';
  %% NON LOCAL MEANS
    

  tic;
  
  threadsPerBlock =[32 32];
  

  k = parallel.gpu.CUDAKernel( 'nonLocalMeansKernel.ptx', ...
                               'nonLocalMeansKernel.cu');


  numberOfBlocks  =ceil( [m n] ./ threadsPerBlock );

  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;

  H=single(gpuArray(H));
  Jgpu=single(gpuArray(J));
  Zgpu = single(zeros([m n], 'gpuArray'));
  If =single(zeros([m n], 'gpuArray'));

  If=gather(feval(k,Jgpu,If,Zgpu,filtSigma*filtSigma,patchSize,m,n,H));

  toc


  %% VISUALIZE RESULT
 
  
  subplot(2,2,1); imagesc(I); title('Original Image')
  subplot(2,2,2); imagesc(J); title('Image with Noise')
  subplot(2,2,3); imagesc(If); title('Filtered Image Cuda')
  subplot(2,2,4); imagesc(If -J); title('Noise')
  colormap gray;
  
   I=single(I);
   squaredErrorImage = (double(I) - double(If)) .^ 2;
   mse = sum(sum(squaredErrorImage)) / (m * n);
   if mse==0
       PSNR=100
   else
   
       PSNR = 10 * log10( 1^2 / mse); %1 epeidh to h megisth timh twn pixel se grayscale einai 1
   end
   ssimval = ssim(If,I);
   messagePass = sprintf('PSNR : %0.3fdB >30 PASS \nMSE  : %0.5f \nSSIM : %0.4f ',PSNR,mse,ssimval);
   messageFail = sprintf('PSNR : %0.3fdB <30 Fail \nMSE  : %0.5f \nSSIM : %0.4f ',PSNR,mse,ssimval);
   if PSNR>30
       msgbox(messagePass);
     else
       msgbox(messageFail);
   end
  
  
  %% (END)

  
  fprintf('...end %s...\n',mfilename);
  
  
  %% filt fix
  
 



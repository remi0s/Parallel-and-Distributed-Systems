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
  %inpute patchSize
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
  if m==64 && patchSize==3
      filtSigma=0.018;
  else
      filtSigma=0.02;
  end
  patchSigma=5/3;
  H = fspecial('gaussian',[patchSize patchSize], patchSigma);
  H = H(:) ./ max(H(:));
  H=H';
  
  
  fprintf('...denoising image...\n ...patchSize=%d MxN= %dx%d ...\n',patchSize,m,n);
  %% NON LOCAL MEANS
  tic;
  bs = [32 32]; % size of the block
  
  % cuda parameters
  threadsPerBlock = bs;
  

  k = parallel.gpu.CUDAKernel( 'nonLocalMeansKernel.ptx', ...
                               'nonLocalMeansKernel.cu');


  numberOfBlocks  =ceil( bs ./ threadsPerBlock );

  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  Zgpu = single(zeros(bs, 'gpuArray'));%float array in GPU
  If =single(zeros(bs, 'gpuArray'));
  H=single(gpuArray(H));



% split array and call cuda

    szJ = size(J);
    nb = szJ ./ bs; % number of blocks in each dimension
    J = mat2cell(J,repmat(bs(1),1,nb(1)), repmat(bs(2),1,nb(2)));%make J into cell array,each sell contains a part of picture
      for i=1:nb(1)
          for j=1:nb(2);
            
           Jsplit = cell2mat(J(i,j));%take one part of the picture
           Jsplit =single(gpuArray(Jsplit));
           %denoising       


            splitIf{i,j}=gather(feval(k,Jsplit,If,Zgpu,filtSigma*filtSigma,patchSize,bs(1),bs(2),H));%call the kernel and save the outcome in splitIf
          end
      end
  toc
    
  If = cell2mat(splitIf); %gather all the splitted arrays into one
  J=cell2mat(J);

  %% VISUALIZE RESULT
  
  subplot(2,2,1); imagesc(I); title('Original Image')
  subplot(2,2,2); imagesc(J); title('Image with Noise')
  subplot(2,2,3); imagesc(If); title('Filtered Image Cuda')
  subplot(2,2,4); imagesc(If -J); title('Noise')
  colormap gray;
  
  %% Result Verification
   I=single(I);
   squaredErrorImage = (double(I) - double(If)) .^ 2;
   mse = sum(sum(squaredErrorImage)) / (m * n);
   if mse==0
       PSNR=100;
   else
   
       PSNR = 10 * log10( 1^2 / mse);%1 because it is the max value of our pixels
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
  

  
 



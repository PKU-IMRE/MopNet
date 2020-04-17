close all;
clear all;

ssim_demoire = 0;
psnr_demoire = 0;
res_ssim = 0;
res_psnr = 0;
for i = 0:11850
    k = i+1;
    i = num2str(i);
    im_source = imread(['./results/d/d',i,'_0_.png']);
    im_source = im2double(im_source);
    im_target = imread(['./results/g/g',i,'_0_.png']);
    im_target = im2double(im_target);
    ssim_demoire = ssim_demoire + ssim(im_source,im_target);
    psnr_demoire = psnr_demoire + psnr(im_source,im_target);
    res_ssim = ssim_demoire / k;
    res_psnr = psnr_demoire / k;
    disp(k)
    disp(res_ssim);
    disp(res_psnr);
end

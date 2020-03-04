%% Load a "clean" image
im_clean = imread('textImage_clean.png');
im_clean = double(im_clean);
[N M] = size(im_clean);

%% Add some noise
noiseStd = 40; % How much white noise to add
im_noisy = im_clean + noiseStd * randn(N,M);

%% Add varying light-intesity model
lightFactor = 70; % Increasing this increases effect of our varying-light model
lightMask = repmat( ((1:M)-M/2)/M, N, 1);
im_light = im_clean + lightFactor * lightMask;

%% Separate background and foreground pixels using our "clean" image
backgroundPixels = im_noisy(im_clean<150);
foregroundPixels = im_noisy(im_clean>150);
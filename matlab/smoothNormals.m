function [smoothed] = smoothNormals(normals, sigma)
% SMOOTHNORMALS Smooth surface normals using Gaussian filtering
%   SMOOTHED = SMOOTHNORMALS(NORMALS, SIGMA) applies Gaussian smoothing
%   to surface normals while preserving their unit length.
%
%   Inputs:
%       NORMALS - H x W x 3 array of surface normals
%       SIGMA - Standard deviation of Gaussian kernel (default: 2.0)
%
%   Output:
%       SMOOTHED - H x W x 3 array of smoothed surface normals
%
%   The function smooths each component separately and then renormalizes.

    if nargin < 2
        sigma = 2.0;
    end
    
    [height, width, ~] = size(normals);
    
    % Create Gaussian kernel
    kernel_size = ceil(sigma * 3) * 2 + 1;  % Ensure odd size
    [x, y] = meshgrid(-floor(kernel_size/2):floor(kernel_size/2), ...
                       -floor(kernel_size/2):floor(kernel_size/2));
    kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    kernel = kernel / sum(kernel(:));
    
    % Smooth each normal component
    smoothed = zeros(size(normals));
    for i = 1:3
        smoothed(:, :, i) = conv2(normals(:, :, i), kernel, 'same');
    end
    
    % Renormalize to unit length
    for i = 1:height
        for j = 1:width
            n = squeeze(smoothed(i, j, :));
            n_mag = norm(n);
            if n_mag > 1e-6
                smoothed(i, j, :) = n / n_mag;
            else
                smoothed(i, j, :) = [0, 0, 1];  % Default upward normal
            end
        end
    end
end

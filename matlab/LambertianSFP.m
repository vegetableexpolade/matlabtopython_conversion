function [normals, albedo] = LambertianSFP(images, light_dirs)
% LAMBERTIANSFP Lambertian Shape from Photometric Stereo
%   [NORMALS, ALBEDO] = LAMBERTIANSFP(IMAGES, LIGHT_DIRS) recovers surface
%   normals and albedo from multiple images under different lighting
%   conditions, assuming Lambertian reflectance.
%
%   Inputs:
%       IMAGES - H x W x N array of N images of the same scene
%       LIGHT_DIRS - N x 3 array of light directions (unit vectors)
%
%   Outputs:
%       NORMALS - H x W x 3 array of surface normals (unit vectors)
%       ALBEDO - H x W array of albedo (diffuse reflectance) values
%
%   The Lambertian reflectance model: I = albedo * max(N' * L, 0)
%   This can be solved using least squares for each pixel.

    [height, width, num_images] = size(images);
    
    % Ensure light directions are unit vectors
    light_dirs = bsxfun(@rdivide, light_dirs, sqrt(sum(light_dirs.^2, 2)));
    
    % Reshape images for processing
    I = reshape(images, height * width, num_images)';
    
    % Initialize outputs
    N = zeros(height * width, 3);
    rho = zeros(height * width, 1);
    
    % Solve for each pixel
    for i = 1:height * width
        intensities = I(:, i);
        
        % Check if pixel has sufficient variation
        if max(intensities) > 0.01 && std(intensities) > 0.001
            % Solve least squares: intensities = light_dirs * (rho * normal)
            % This gives us g = rho * normal
            g = light_dirs \ intensities;
            
            % Extract albedo and normal
            rho(i) = norm(g);
            
            if rho(i) > 1e-6
                N(i, :) = g' / rho(i);
            else
                N(i, :) = [0, 0, 1];  % Default upward normal
            end
        else
            % Insufficient data, use default
            N(i, :) = [0, 0, 1];
            rho(i) = 0;
        end
    end
    
    % Reshape outputs
    normals = reshape(N, height, width, 3);
    albedo = reshape(rho, height, width);
    
    % Ensure normals are unit vectors
    for i = 1:height
        for j = 1:width
            n_mag = norm(squeeze(normals(i, j, :)));
            if n_mag > 1e-6
                normals(i, j, :) = normals(i, j, :) / n_mag;
            end
        end
    end
end

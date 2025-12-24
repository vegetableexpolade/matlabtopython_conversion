function light_dirs = findLight(images, normals)
% FINDLIGHT Estimate light source directions from images
%   LIGHT_DIRS = FINDLIGHT(IMAGES, NORMALS) estimates the light source
%   directions for a set of images given known surface normals.
%
%   Inputs:
%       IMAGES - H x W x N array where N is the number of images
%       NORMALS - H x W x 3 array of surface normals (or M x 3 where M = H*W)
%
%   Output:
%       LIGHT_DIRS - N x 3 array of light directions (unit vectors)
%
%   This function uses least squares to estimate light directions assuming
%   Lambertian reflectance: I = rho * max(N' * L, 0)

    [height, width, num_images] = size(images);
    
    % Reshape images to column vectors
    I = reshape(images, height * width, num_images);
    
    % Handle normals input
    if ndims(normals) == 3
        N = reshape(normals, height * width, 3);
    else
        N = normals;
    end
    
    % Remove invalid pixels (where normal magnitude is too small)
    valid_mask = sqrt(sum(N.^2, 2)) > 0.1;
    I_valid = I(valid_mask, :);
    N_valid = N(valid_mask, :);
    
    % Normalize normals
    N_norm = bsxfun(@rdivide, N_valid, sqrt(sum(N_valid.^2, 2)));
    
    % Initialize light directions
    light_dirs = zeros(num_images, 3);
    
    % Estimate each light direction independently
    for i = 1:num_images
        intensities = I_valid(:, i);
        
        % Remove pixels with very low intensity
        bright_mask = intensities > max(intensities) * 0.1;
        
        if sum(bright_mask) > 10
            N_bright = N_norm(bright_mask, :);
            I_bright = intensities(bright_mask);
            
            % Solve for light direction using least squares
            % I = (N' * L) * rho, we solve for L assuming rho is absorbed in L
            light_dir = N_bright \ I_bright;
            
            % Normalize light direction
            light_dirs(i, :) = light_dir' / norm(light_dir);
        else
            % Default to upward pointing light if insufficient data
            light_dirs(i, :) = [0, 0, 1];
        end
    end
end
